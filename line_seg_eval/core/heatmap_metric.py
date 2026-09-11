"""Heatmap-based line metric: rasterizes lines and scores overlapping pixels."""
import numpy as np

from collections import defaultdict
from line_seg_eval import _C


class LINEeval_heatmap:
    """Computes heatmap average precision (APh) and peak F-score (Fh) per class.

    Rather than matching whole segments, each predicted and ground-truth line is
    rasterized and scored pixel by pixel, so partially correct lines earn partial credit.
    Per-image statistics accumulate across `update` calls and are reduced in `summarize`.
    """

    def __init__(self):
        """Creates the C++ rasterizing matcher and clears the accumulators."""
        self.matcher = _C.HeatmapMatcher()

        self.fixed_thresholds = [
            0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.525, 0.55, 0.575, 
            0.6, 0.625, 0.65, 0.675, 0.7, 0.8, 0.9, 0.95, 
            0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999
        ]

        self.reset()

    def reset(self):
        """Clears the per-class pixel statistics, readying a fresh evaluation."""
        # Store results globally across all images
        # Structure: { class_id: { 'dt_stats': [(score, tp_pix, fp_pix), ...], 'total_gt_pix': 0 } }
        self.class_results = defaultdict(lambda: {'dt_stats': [], 'total_gt_pix': 0})
        self.stats = {}

    def update(self, dt_lines, dt_scores, dt_labels, gt_lines, gt_labels, height, width):
        """Rasterizes one image's lines and records per-prediction pixel hit counts.

        Called once per image, since rasterization needs that image's dimensions. Missing
        labels are treated as all class 0.

        Args:
            dt_lines: (N, 2, 2) predicted endpoints in (y, x) order, in image pixels
            dt_scores: (N,) confidence scores, descending
            dt_labels: (N,) predicted class labels; may be empty
            gt_lines: (M, 2, 2) ground-truth endpoints in the same form
            gt_labels: (M,) ground-truth class labels; may be empty
            height: image height in pixels
            width: image width in pixels
        """
        # Note: The 'update' in LineEvaluator is called per-image inside a loop.
        # So 'dt_lines' here corresponds to ONE image.
        
        # 1. Identify Classes Present (Usually just class 0)
        if len(gt_labels) == 0: gt_labels = np.zeros(len(gt_lines), dtype=np.int32)
        if len(dt_labels) == 0: dt_labels = np.zeros(len(dt_lines), dtype=np.int32)
        
        unique_classes = set(np.unique(gt_labels)) | set(np.unique(dt_labels))

        for cls in unique_classes:
            # Filter Data for this Class
            g_mask = (gt_labels == cls)
            d_mask = (dt_labels == cls)

            cls_gt_lines = gt_lines[g_mask]
            cls_dt_lines = dt_lines[d_mask]
            cls_dt_scores = dt_scores[d_mask]

            # C++ Rasterization (Per Image)
            # Returns: (tp_counts_array, fp_counts_array, total_gt_pixels_int)
            tp_counts, fp_counts, total_pixels = self.matcher.evaluate_sequence(cls_dt_lines, cls_gt_lines, height, width)
            
            # Store Statistics
            # We pack (score, tp, fp) for every predicted line
            stats = np.stack((cls_dt_scores, tp_counts, fp_counts), axis=1)
            
            self.class_results[cls]['dt_stats'].append(stats)
            self.class_results[cls]['total_gt_pix'] += total_pixels

    def accumulate(self):
        """No-op: statistics are already accumulated as each image is processed."""
        # Data is already accumulated in `self.class_results` during update
        pass
    
    def summarize(self):
        """Builds each class's precision-recall curve, prints APh and Fh, and stores them.

        Results land in `self.stats` under 'AP' and 'F', averaged over classes, which is
        what the validator reads.
        """
        print("\n" + "="*50)
        print(f"{'Class':<10} | {'APh':<10} | {'Fh':<10}")
        print("-" * 50)
        
        aps = []
        fhs = []

        # Iterate PER CLASS
        for cls in sorted(self.class_results.keys()):
            data = self.class_results[cls]
            total_gt = data['total_gt_pix']
            
            if total_gt == 0 or not data['dt_stats']: 
                # print(f"{cls:<10} | {0.0:<10.1f} | {0.0:<10.1f}")
                continue

            all_stats = np.concatenate(data['dt_stats'], axis=0)
            
            # 1. Sort All Predictions for this Class (Implicitly checking ALL thresholds)
            sort_idx = np.argsort(-all_stats[:, 0], kind='mergesort')
            sorted_stats = all_stats[sort_idx]
            
            # 2. Cumulative Sums
            tp_cum = np.cumsum(sorted_stats[:, 1])
            fp_cum = np.cumsum(sorted_stats[:, 2])

            # 3. Compute Curve
            recall = tp_cum / total_gt
            precision = tp_cum / (tp_cum + fp_cum + 1e-9)

            # 4. Compute Single Metrics
            ap_h = self._auc(recall, precision) * 100.0
            
            f1_curve = 2 * (precision * recall) / (precision + recall + 1e-9)
            fh = np.max(f1_curve) * 100.0 if len(f1_curve) > 0 else 0.0

            aps.append(ap_h)
            fhs.append(fh)

            print(f"{cls:<10} | {ap_h:<10.1f} | {fh:<10.1f}")
        
        print("-" * 50)
        if aps:
            mean_ap = float(np.mean(aps))
            mean_fh = float(np.mean(fhs))
            print(f"{'MEAN':<10} | {mean_ap:<10.1f} | {mean_fh:<10.1f}")
            self.stats['AP'] = mean_ap
            self.stats['F'] = mean_fh
        else:
            print(f"{'MEAN':<10} | {0.0:<10.1f} | {0.0:<10.1f}")
            self.stats['AP'] = 0.0
            self.stats['F'] = 0.0
        print("="*50 + "\n")

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
