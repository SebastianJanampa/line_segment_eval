# Line Segment Evaluation

A high-performance, C++ optimized library for evaluating line segment detection models. This library serves as a drop-in evaluation engine that computes **Structural Average Precision (sAP)** and **F1 Scores** significantly faster than pure Python implementations.

Designed with a modular architecture similar to `pycocotools`, it supports batch processing, incremental updates, and global accumulation of metrics.

## 🚀 Features

* **C++ Backend:** Core matching logic (`Greedy Match`) and distance calculations are implemented in C++14 using `pybind11` for maximum speed.
* **Metric Support:**
    * **sAP (Structural Average Precision):** Calculated at thresholds 5, 10, and 15.
    * **F1 Score:** Reports the maximum possible F1 score across all confidence thresholds.
    * **APh & Fh (Heatmap Average Precision & F1):** Exact pixel-level evaluation matching the official DT-LSD/L-CNN benchmark behavior, utilizing a 1% image diagonal spatial tolerance.
* **Framework Agnostic:** Works directly with NumPy arrays but includes built-in helpers for PyTorch tensors (auto-detach/CPU conversion).
* **Memory Efficient:** Processes batches incrementally; does not store heavy coordinate tensors in memory.

## 🛠️ Installation

### Prerequisites
* Python ≥ 3.8
* C++ Compiler (GCC, Clang, or MSVC) supporting C++14
* NumPy
* PyTorch (Optional, but recommended)

### Build from Pypi
```bash
pip install line-seg-eval
```

### Build from Source
```bash
git clone https://github.com/SebastianJanampa/line_segment_eval.git
cd line-segment-eval

# Install in editable mode (recommended for development)
pip install -e .
```

## 💻 Usage

### Basic Example
The library uses a **Controller/Worker** pattern. 
You instantiate the `LineEvaluator`,
update it with batches of predictions 
and ground truths, and finally summarize the results.
```python
from line_seg_eval.pytorch import LineEvaluator

# 1. Initialize
# metrics=['lines'] enables sAP and F1 calculation
evaluator = LineEvaluator(metrics=['endpoints', 'heatmap'], 
    do_postprocess=True, 
    nms_thresh=0.01
)

# 2. Training/Validation Loop
for batch in dataloader:
    predictions = model(batch['image'])  # Your model output
    targets = batch['targets']           # Ground truth list
    
    # 3. Update (Process batch immediately in C++)
    # predictions: dict with 'lines' and 'scores'
    # targets: list of dicts with 'lines'
    evaluator.update(predictions, targets)

# 4. End of Epoch
evaluator.accumulate()  # Global sort and merge
evaluator.summarize()   # Print table of results
```
### Expected Input Format

**Predictions (`dict`):**

| Key | Shape | Description |
| :--- | :--- | :--- |
| `lines` / `pred_lines` | `[B, N, 2, 2]` or `[B, N, 4]` | Predicted line segments (x1, y1, x2, y2) |
| `scores` / `pred_logits` | `[B, N]` | Confidence scores (raw logits or probabilities) |
| `labels` / `pred_labels` | `[B, N]` | (Optional) Class labels for multi-class evaluation |

**Ground Truths (`list` of `dict`):**
A list where each item corresponds to one image in the batch.

| Key | Shape | Description |
| :--- | :--- | :--- |
| `lines` | `[M, 2, 2]` or `[M, 4]` | Ground truth segments |
| `labels`| `[M]` | (Optional) Ground truth class labels |
| `height`| `int` | **Required for Heatmap:** Original image height (e.g., 512) |
| `width` | `int` | **Required for Heatmap:** Original image width (e.g., 512) |

*Note: Coordinates are automatically scaled and flipped geometrically within the library to match standard benchmarks. Endpoints are evaluated at a fixed 128x128 scale, while Heatmap metrics are evaluated at the original image resolution (`height` and `width`).*

## 📊 Metrics Explained

### Structural Metrics (sAP / sF)
Evaluates the geometric distance between line endpoints. Structural metrics are evaluated at a fixed **128x128** spatial resolution, regardless of the original image size. 
* **sAP5:** Strict match (Distance < 5)
* **sAP10:** Standard match (Distance < 10)
* **sAP15:** Loose match (Distance < 15)
* **sF:** Reports the maximum possible F1 score across all confidence thresholds for the structural matches.

### Heatmap Metrics (APh / Fh)
Evaluates lines at the pixel level. Both ground truth and predicted lines are rasterized into mathematical 2D grids at their **original image resolution** (e.g., 512x512).
* **1% Spatial Tolerance:** A predicted pixel is considered a True Positive if it falls within a dynamic radius of `0.01 * sqrt(H^2 + W^2)` from a ground truth pixel.
* **Collinear Clipping (NMS):** When `do_postprocess=True` is enabled, the library applies the exact Collinear Line Clipping algorithm used in classic literature benchmarks (like L-CNN and DT-LSD) to aggressively merge redundant, overlapping predictions before evaluation.
* **APh:** Heatmap Area Under the Curve (AUC).
* **Fh:** Maximum possible F1 score for the pixel-level heatmap matches.
