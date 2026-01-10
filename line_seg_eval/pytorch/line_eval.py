from line_seg_eval import LINEeval

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

                # 2. Prepare Data (Scale, Flip, Sort)
                gt_lines, _ = _prepare_lines(raw_gt)
                dt_lines, dt_scores = _prepare_lines(raw_dt, raw_scores)

                if len(dt_lines) > 0:
                    # 3. Dispatch to Worker
                    self.evaluators['lines'].update(dt_lines, dt_scores, gt_lines)

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