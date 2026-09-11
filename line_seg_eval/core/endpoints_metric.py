"""Endpoint-based line metrics: structural average precision (sAP) and F-score."""
import numpy as np

from collections import defaultdict
from line_seg_eval import _C


class LINEeval_endpoints:
    """Computes structural AP and F-score by matching whole segments to ground truth.

    A prediction counts as correct when the squared distance between its endpoints and a
    ground-truth line's falls under a threshold, so several thresholds give sAP5, sAP10
    and so on. Matching runs in C++; this class accumulates the hits and reduces them.
    """

    def __init__(self, thresholds=[5, 10, 15]):
        """Creates the C++ matcher and clears the accumulators.

        Args:
            thresholds: endpoint distance thresholds to evaluate at
        """
        self.matcher = _C.LineMatcher()
        self.thresholds = thresholds
        self.reset()

    def reset(self):
        """Clears the accumulated matches, scores and ground-truth counts."""
        # Raw Data Storage
        self.matches = {str(t): {'tp': [], 'fp': []} for t in self.thresholds}
        self.scores = []
        self.dt_labels = [] # Store prediction label

        # GT Counts
        self.total_gt = 0
        self.gt_counts_per_class = defaultdict(int)

        # Computed Data
        self.eval_data = {}
        self.stats = {}

    def update(self, dt_lines, dt_scores, dt_labels, gt_lines, gt_labels):
        """Matches one image's predictions against its ground truth, at every threshold.

        The C++ matcher enforces that a match shares the same class. Missing labels are
        treated as all class 0.

        Args:
            dt_lines: (N, 2, 2) predicted endpoints in (y, x) order
            dt_scores: (N,) confidence scores, descending
            dt_labels: (N,) predicted class labels; may be empty
            gt_lines: (M, 2, 2) ground-truth endpoints in the same form
            gt_labels: (M,) ground-truth class labels; may be empty
        """
        # 1. Update GT Counts Per Class
        if len(gt_labels) > 0:
            unique, counts = np.unique(gt_labels, return_counts=True)
            for u, c in zip(unique, counts):
                self.gt_counts_per_class[u] += c
        else:
            # Fallback for binary (no labels passed)
            self.gt_counts_per_class[0] += len(gt_lines)
            if len(dt_labels) == 0:
                dt_labels = np.zeros(len(dt_lines), dtype=np.int32)
            if len(gt_labels) == 0:
                gt_labels = np.zeros(len(gt_lines), dtype=np.int32)

        # 2. Run C++ Matcher
        batch_res = self.matcher.match_lines(
            dt_lines, gt_lines,
            dt_labels, gt_labels,
            self.thresholds
        )

        # 3. Store Results
        for k, (tp, fp) in batch_res.items():
            self.matches[k]['tp'].append(tp)
            self.matches[k]['fp'].append(fp)

        self.scores.append(dt_scores)
        self.dt_labels.append(dt_labels)
        self.total_gt += len(gt_lines)

    def accumulate(self):
        """Concatenates every image's matches and sorts them by descending score.

        The global sort is what makes the precision-recall curve meaningful across images
        rather than within each one.
        """
        if not self.scores:
            print("LINEeval: No predictions to accumulate.")
            return

        # 1. Concatenate everything
        scores = np.concatenate(self.scores)
        labels = np.concatenate(self.dt_labels)

        # 2. Global Sort Indices
        sort_inds = np.argsort(-scores, kind='mergesort')

        # 3. Store Sorted Data
        self.eval_data = {}

        # We store the sorted LABELS so we can filter by class later
        self.eval_data['labels'] = labels[sort_inds]

        for k in self.matches:
            if not self.matches[k]['tp']:
                continue
            tp = np.concatenate(self.matches[k]['tp'])[sort_inds]
            fp = np.concatenate(self.matches[k]['fp'])[sort_inds]
            self.eval_data[k] = {'tp': tp, 'fp': fp}

    def summarize(self):
        """Prints sAP and sF per class and threshold, and stores the class means.

        Results land in `self.stats` under 'sAP_<t>' and 'sF_<t>', which is what the
        validator reads.
        """
        unique_classes = sorted(self.gt_counts_per_class.keys())

        # Header
        print("\n" + "=" * 95)
        header = f"{'Class':<10} | "
        for t in self.thresholds: header += f"sAP{t:<3} "
        header += "| "
        for t in self.thresholds: header += f"sF{t:<4} "
        print(header)
        print("-" * 95)

        # Storage for Mean calculation
        aps_per_thresh = {t: [] for t in self.thresholds}
        sfs_per_thresh = {t: [] for t in self.thresholds}

        # --- 1. Per-Class Loop ---
        for cls_id in unique_classes:
            row_str = f"{cls_id:<10} | "

            # Filter mask for this class
            cls_mask = (self.eval_data['labels'] == cls_id)
            gt_count = self.gt_counts_per_class[cls_id]

            if gt_count == 0:
                continue# Compute sAP for all thresholds
            for t in self.thresholds:
                k = str(t)
                val = 0.0
                if k in self.eval_data:
                    tp = self.eval_data[k]['tp'][cls_mask]
                    fp = self.eval_data[k]['fp'][cls_mask]
                    val = self._calc_ap(tp, fp, gt_count)

                aps_per_thresh[t].append(val)
                row_str += f"{val:<6.1f} "

            row_str += "| "

            # Compute sF for all thresholds
            for t in self.thresholds:
                k = str(t)
                val = 0.0
                if k in self.eval_data:
                    tp = self.eval_data[k]['tp'][cls_mask]
                    fp = self.eval_data[k]['fp'][cls_mask]
                    val = self._calc_sf(tp, fp, gt_count)

                sfs_per_thresh[t].append(val)
                row_str += f"{val:<6.1f} "

            print(row_str)

        print("-" * 95)

        # --- 2. Mean (General) Metrics ---
        mean_row = f"{'MEAN':<10} | "

        # Mean sAP
        for t in self.thresholds:
            m_ap = np.mean(aps_per_thresh[t]) if aps_per_thresh[t] else 0.0
            mean_row += f"{m_ap:<6.1f} "
            self.stats[f'sAP_{t}'] = float(m_ap)

        mean_row += "| "

        # Mean sF
        for t in self.thresholds:
            m_sf = np.mean(sfs_per_thresh[t]) if sfs_per_thresh[t] else 0.0
            mean_row += f"{m_sf:<6.1f} "
            self.stats[f'sF_{t}'] = float(m_sf)

        print(mean_row)
        print("="*95 + "\n")

    def _calc_ap(self, tp, fp, total_gt):
        """Computes average precision from sorted true and false positive flags.

        Args:
            tp: true-positive flags in descending score order
            fp: false-positive flags in the same order
            total_gt: number of ground-truth lines for this class

        Returns:
            ap: average precision as a percentage, or 0.0 when there are no predictions
        """
        if len(tp) == 0: return 0.0
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / total_gt
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)
        return self._auc(recall, precision) * 100.0

    def _calc_sf(self, tp, fp, total_gt):
        """Computes the peak F-score along the precision-recall curve.

        Args:
            tp: true-positive flags in descending score order
            fp: false-positive flags in the same order
            total_gt: number of ground-truth lines for this class

        Returns:
            sf: the best F-score as a percentage, or 0.0 when there are no predictions
        """
        if len(tp) == 0: return 0.0
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / total_gt
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)

        # sF = 2 * (P * R) / (P + R)
        f_curve = 2 * (precision * recall) / (precision + recall + 1e-9)
        return np.max(f_curve) * 100.0

    def _auc(self, recall, precision):
        """Computes average precision as the area under an interpolated PR curve.

        Precision is made monotonically decreasing before integrating, the standard
        all-points interpolation.

        Args:
            recall: recall values, ascending
            precision: precision values matching `recall`

        Returns:
            ap: the area under the interpolated curve, in [0, 1]
        """
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        i = np.where(mrec[1:] != mrec[:-1])[0]
        return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])