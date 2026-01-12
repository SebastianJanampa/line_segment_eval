try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False

from line_seg_eval import LINEeval

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
    def __init__(self, metrics=['lines', 'masks']):
        """
        metrics: list of data types to evaluate.
        """
        self.evaluators = {}

        # Initialize Workers based on requested types
        if 'lines' in metrics:
            self.evaluators['lines'] = LINEeval(thresholds=[5, 10, 15])

        if 'masks' in metrics:
            # Future expansion
            # self.evaluators['masks'] = COCOeval_masks(...)
            pass

    def reset(self):
        for evaluator in self.evaluators.values():
            evaluator.reset()

    def update(self, predictions, ground_truths):
        """
        Updates the internal state with a new batch of data.
        """
        batch_size = len(ground_truths)

        # Check which keys to process
        process_lines = 'lines' in self.evaluators

        for i in range(batch_size):
            gt_item = ground_truths[i]

            # --- LINES HANDLING ---
            if process_lines:
                # 1. Extract Raw Data
                raw_gt = gt_item['lines']
                raw_gt_labels = gt_item.get('labels', None)

                # Handle dictionary naming variations
                if 'pred_lines' in predictions:
                    raw_dt = predictions['pred_lines'][i]
                else:
                    raw_dt = predictions['lines'][i]  # Fallback

                if 'scores' in predictions:
                    raw_scores = predictions['scores'][i]
                elif 'pred_logits' in predictions:
                    raw_scores = predictions['pred_logits'][i]
                else:
                    raise ValueError("Predictions missing 'scores' or 'pred_logits'")

                raw_dt_labels = None
                if 'labels' in predictions:
                    raw_dt_labels = predictions['labels'][i]
                elif 'pred_classes' in predictions:
                    raw_dt_labels = predictions['pred_classes'][i]

                # 2. Prepare
                gt_lines, _, gt_labels = _prepare_data(raw_gt, None, raw_gt_labels)
                dt_lines, dt_scores, dt_labels = _prepare_data(raw_dt, raw_scores, raw_dt_labels)

                if len(dt_lines) > 0:
                    # 3. Dispatch to Worker
                    self.evaluators['lines'].update(dt_lines, dt_scores, dt_labels, gt_lines, gt_labels)

            # --- MASKS HANDLING (Future) ---
            # if 'masks' in self.evaluators:
            #     ... extract and dispatch masks ...

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