try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False
import numpy as np

from line_seg_eval import LINEeval_heatmap, LINEeval_endpoints
from line_seg_eval import _C

def _to_numpy(data):
    if _has_torch and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)

def _prepare_data(lines, scores=None, labels=None):
    """
    Standardizes inputs:
    1. Converts to NumPy
    2. Reshapes Lines [N, 2, 2]
    3. Scales/Flips Lines
    4. Sorts ALL arrays by Score (Descending)
    """
    # --- 1. Lines ---
    lines = _to_numpy(lines)
    if lines.ndim == 2: lines = lines.reshape(-1, 2, 2)
    lines = lines[..., ::-1] * 128.0  # Scale & Flip

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
    def __init__(self, metrics=['endpoints', 'heatmap'], do_postprocess=True, nms_thresh=0.01):
        """
        metrics: list of data types to evaluate.
        """
        self.evaluators = {}
        self.do_postprocess = do_postprocess
        self.nms_thresh = nms_thresh

        # Initialize Workers based on requested types
        if 'endpoints' in metrics:
            self.evaluators['endpoints'] = LINEeval_endpoints(thresholds=[5, 10, 15])

        if 'heatmap' in metrics:
            # Future expansion
            img_size = 128 # this is the default in many line segment detectors
            self.evaluators['heatmap'] = LINEeval_heatmap()

    def reset(self):
        for evaluator in self.evaluators.values():
            evaluator.reset()

    def update(self, predictions, ground_truths):
        """
        Updates the internal state with a new batch of data.
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
            gt_lines_128, _, gt_labels = _prepare_data(raw_gt, None, raw_gt_labels)
            dt_lines_128, dt_scores, dt_labels = _prepare_data(raw_dt, raw_scores, raw_dt_labels)

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
                        gt_lines_hm[:, :, 0] *= (h / 128.0)  # Y
                        gt_lines_hm[:, :, 1] *= (w / 128.0)  # X
                        dt_lines_hm[:, :, 0] *= (h / 128.0)  # Y
                        dt_lines_hm[:, :, 1] *= (w / 128.0)  # X
                        
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
        """
        Delegates accumulation to all registered workers.
        """
        for key, evaluator in self.evaluators.items():
            print(f"Accumulating {key}...")
            evaluator.accumulate()

    def summarize(self):
        """
        Delegates summarization to all registered workers.
        """
        for key, evaluator in self.evaluators.items():
            print(f"\nEvaluation Summary: {key.upper()}")
            evaluator.summarize()
