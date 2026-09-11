"""LineEvaluator: converts model outputs into the form each metric expects."""
try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False
import numpy as np

from line_seg_eval import LINEeval_heatmap, LINEeval_endpoints
from line_seg_eval import _C

def _to_numpy(data):
    """Converts a torch tensor or array-like to a NumPy array.

    Args:
        data: a torch tensor, NumPy array, or anything np.asarray accepts

    Returns:
        array: the data as a NumPy array, detached and moved to CPU if it was a tensor
    """
    if _has_torch and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)

def _prepare_data(lines, scores=None, labels=None, img_size=128.0):
    """Normalizes lines, scores and labels into the layout the metrics expect.

    Converts to NumPy, reshapes lines to (N, 2, 2), swaps each point to (y, x) order and
    scales to `img_size`. When scores are given, all three arrays are sorted by descending
    score, which the average-precision metrics rely on.

    Args:
        lines: (N, 4) or (N, 2, 2) line endpoints normalized to [0, 1]
        scores: (N,) confidence scores; None marks the input as ground truth, which is
            returned unsorted
        labels: (N,) class labels, optional
        img_size: side length the normalized coordinates are scaled to

    Returns:
        lines: (N, 2, 2) endpoints in (y, x) order, scaled to `img_size`
        scores: (N,) scores in descending order, or None for ground truth
        labels: (N,) labels in the matching order, empty when none were given
    """
    # --- 1. Lines ---
    lines = _to_numpy(lines)
    if lines.ndim == 2: lines = lines.reshape(-1, 2, 2)
    lines = lines[..., ::-1] * img_size  # Scale & Flip

    if scores is None:# For Ground Truths: Just return lines and labels
        clean_labels = np.array([], dtype=np.int32)
        if labels is not None:
            clean_labels = _to_numpy(labels).astype(np.int32)
        return lines, None, clean_labels

    # --- 2. Scores ---
    scores = _to_numpy(scores)
    if scores.ndim > 1:
        scores = scores[..., 0]

    # --- 3. Sorting ---
    # We must sort lines and labels based on the score order
    idx = np.argsort(-scores, kind='mergesort')

    sorted_lines = lines[idx]
    sorted_scores = scores[idx]

    # --- 4. Labels (Optional) ---
    sorted_labels = np.array([], dtype=np.int32)
    if labels is not None:
        labels = _to_numpy(labels).astype(np.int32)
        sorted_labels = labels[idx]

    return sorted_lines, sorted_scores, sorted_labels

class LineEvaluator:
    """Runs several line-segment metrics over a stream of prediction batches.

    Owns one worker per requested metric and feeds each the representation it needs: the
    endpoint metrics score raw coordinates at `img_size`, while the heatmap metric scores
    coordinates rescaled to the original image and optionally NMS-clipped.
    """

    def __init__(self, metrics=['endpoints', 'heatmap'], do_postprocess=True, nms_thresh=0.01, img_size=128.0):
        """Builds one worker per requested metric.

        Args:
            metrics: which metrics to compute, any of 'endpoints' and 'heatmap'
            do_postprocess: whether to run the C++ NMS-style clipping before the heatmap
                metric
            nms_thresh: clipping threshold as a fraction of the image diagonal
            img_size: side length normalized coordinates are scaled to
        """
        self.evaluators = {}
        self.do_postprocess = do_postprocess
        self.nms_thresh = nms_thresh
        self.img_size = img_size

        # Initialize Workers based on requested types
        if 'endpoints' in metrics:
            self.evaluators['endpoints'] = LINEeval_endpoints(thresholds=[5, 10, 15])

        if 'heatmap' in metrics:
            # Future expansion
            img_size = 128 # this is the default in many line segment detectors
            self.evaluators['heatmap'] = LINEeval_heatmap()

    def reset(self):
        """Clears the accumulated state of every metric, readying a fresh evaluation."""
        for evaluator in self.evaluators.values():
            evaluator.reset()

    def update(self, predictions, ground_truths):
        """Feeds one batch of predictions and targets to every metric.

        Accepts either naming convention for the prediction keys, so raw model output and
        postprocessed output both work. Images with no surviving predictions are skipped.

        Args:
            predictions: dict holding lines under 'lines' or 'pred_lines', scores under
                'scores' or 'pred_logits', and labels under 'labels' or 'pred_labels',
                each indexable by batch position
            ground_truths: list of per-image target dicts with 'lines', and optionally
                'labels' and 'size'

        Raises:
            ValueError: if the predictions dict carries neither name for lines, scores or
                labels
        """
        batch_size = len(ground_truths)

        # Check which keys to process

        for i in range(batch_size):
            gt_item = ground_truths[i]

            # --- LINES HANDLING ---
            # 1. Extract Raw Data
            raw_gt = gt_item['lines']
            raw_gt_labels = gt_item.get('labels', None)

            # Handle dictionary naming variations
            if 'lines' in predictions:
                raw_dt = predictions['lines'][i]  # Fallback
            elif 'pred_lines' in predictions:
                raw_dt = predictions['pred_lines'][i]
            else:
                raise ValueError("Predictions missing 'lines' or 'pred_lines'")


            if 'scores' in predictions:
                raw_scores = predictions['scores'][i]
            elif 'pred_logits' in predictions:
                raw_scores = predictions['pred_logits'][i]
            else:
                raise ValueError("Predictions missing 'scores' or 'pred_logits'")

            if 'labels' in predictions:
                raw_dt_labels = predictions['labels'][i]
            elif 'pred_labels' in predictions:
                raw_dt_labels = predictions['pred_labels'][i]
            else:
                raise ValueError("Predictions missing 'labels' or 'pred_labels'")

            # 2. Prepare
            gt_lines_128, _, gt_labels = _prepare_data(raw_gt, None, raw_gt_labels, img_size=self.img_size)
            dt_lines_128, dt_scores, dt_labels = _prepare_data(raw_dt, raw_scores, raw_dt_labels, img_size=self.img_size)

            if len(dt_lines_128) == 0:
                continue
                
            for metric in self.evaluators:
                # 3. Dispatch to Worker
                if metric == 'heatmap':
                    # Extract original dimensions (fallback to 128)
                    h, w = _to_numpy(gt_item.get('size', [128, 128]))
                    
                    # Create isolated copies for the heatmap
                    gt_lines_hm = gt_lines_128
                    dt_lines_hm = dt_lines_128
                    dt_scores_hm = dt_scores

                    # Scale to Real Image Dimensions
                    if len(dt_lines_hm) > 0:
                        gt_lines_hm[:, :, 0] *= (h / self.img_size)  # Y
                        gt_lines_hm[:, :, 1] *= (w / self.img_size)  # X
                        dt_lines_hm[:, :, 0] *= (h / self.img_size)  # Y
                        dt_lines_hm[:, :, 1] *= (w / self.img_size)  # X
                        
                    # Apply C++ Postprocessing (Clipping) EXCLUSIVELY for Heatmap
                    if self.do_postprocess and len(dt_lines_hm) > 0:
                        diag_real = np.sqrt(h**2 + w**2)
                        clip_thresh = diag_real * self.nms_thresh
                        
                        dt_lines_hm, dt_scores_hm = _C.postprocess(
                            dt_lines_hm.astype(np.float32), 
                            dt_scores_hm.astype(np.float32), 
                            clip_thresh, 0, False
                        )
                        # Create generic labels since postprocess reshapes the arrays
                        dt_labels_hm = np.zeros(len(dt_lines_hm), dtype=np.int32)
                    else:
                        dt_labels_hm = dt_labels
                    
                    self.evaluators[metric].update(
                            dt_lines_hm, dt_scores_hm, dt_labels_hm, 
                            gt_lines_hm, gt_labels, h, w
                        )

                elif metric == 'endpoints':
                    # sAP uses the RAW, unclipped 128x128 representations
                    self.evaluators[metric].update(
                        dt_lines_128, dt_scores, dt_labels, 
                        gt_lines_128, gt_labels
                    )
                #self.evaluators[metric].update(dt_lines, dt_scores, dt_labels, gt_lines, gt_labels)

    def accumulate(self):
        """Tells every metric to turn its accumulated batches into final statistics."""
        for key, evaluator in self.evaluators.items():
            print(f"Accumulating {key}...")
            evaluator.accumulate()

    def summarize(self):
        """Prints each metric's results, in the metric's own format."""
        for key, evaluator in self.evaluators.items():
            print(f"\nEvaluation Summary: {key.upper()}")
            evaluator.summarize()
