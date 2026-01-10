import numpy as np
from line_seg_eval import _C


class LINEeval:
    def __init__(self, thresholds=[5, 10, 15]):
        """
        Worker class for Line Evaluation.
        """
        self.matcher = _C.LineMatcher()
        self.thresholds = thresholds
        self.reset()

    def reset(self):
        self.matches = {str(t): {'tp': [], 'fp': []} for t in self.thresholds}
        self.scores = []
        self.total_gt = 0
        self.stats = {}
        self.eval_data = {}

    def update(self, dt_lines, dt_scores, gt_lines):
        batch_res = self.matcher.match_lines(dt_lines, gt_lines, self.thresholds)

        for k, (tp, fp) in batch_res.items():
            self.matches[k]['tp'].append(tp)
            self.matches[k]['fp'].append(fp)

        self.scores.append(dt_scores)
        self.total_gt += len(gt_lines)

    def accumulate(self):
        if not self.scores:
            print("LINEeval: No predictions to accumulate.")
            return

        scores = np.concatenate(self.scores)
        sort_inds = np.argsort(-scores, kind='mergesort')

        self.eval_data = {}
        for k in self.matches:
            if not self.matches[k]['tp']:
                continue
            tp = np.concatenate(self.matches[k]['tp'])[sort_inds]
            fp = np.concatenate(self.matches[k]['fp'])[sort_inds]
            self.eval_data[k] = {'tp': tp, 'fp': fp}

    def summarize(self):
        """
        1. Calculates all metrics.
        2. Prints all sAP values.
        3. Prints all F1 values.
        """
        # --- 1. Calculation Pass ---
        for t in self.thresholds:
            k = str(t)

            # Default to 0.0 if no data
            if k not in self.eval_data:
                self.stats[f'sAP{k}'] = 0.0
                self.stats[f'F1@{k}'] = 0.0
                continue

            # Retrieve Data
            tp = self.eval_data[k]['tp']
            fp = self.eval_data[k]['fp']

            # Compute Curves
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / self.total_gt
            precision = tp_cum / (tp_cum + fp_cum + 1e-9)

            # Store Results
            self.stats[f'sAP{k}'] = self._compute_sap(recall, precision)
            self.stats[f'F1@{k}'] = self._compute_f1(recall, precision)

        # --- 2. Print Pass: sAP ---
        print("-" * 30)
        print("sAP Results")
        print("-" * 30)
        for t in self.thresholds:
            k = str(t)
            val = self.stats.get(f'sAP{k}', 0.0)
            print(f"sAP@{k:<4}: {val:.1f}")

        # --- 3. Print Pass: F1 ---
        print("-" * 30)
        print("sF Results")
        print("-" * 30)
        for t in self.thresholds:
            k = str(t)
            val = self.stats.get(f'F1@{k}', 0.0)
            print(f"sF@{k:<4}: {val:.1f}")
        print("-" * 30)

    def _compute_sap(self, recall, precision):
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        i = np.where(mrec[1:] != mrec[:-1])[0]
        return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]) * 100.0

    def _compute_f1(self, recall, precision):
        f1_curve = 2 * (precision * recall) / (precision + recall + 1e-9)
        return np.max(f1_curve) * 100.0