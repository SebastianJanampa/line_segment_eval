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
cd line_segment_eval

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
# 'endpoints' enables sAP and sF; 'heatmap' enables APh and Fh
evaluator = LineEvaluator(metrics=['endpoints', 'heatmap'], 
    do_postprocess=True, 
    nms_thresh=0.01,
    img_size=128.0
)

# 2. Training/Validation Loop
evaluator.reset()  # Clear state from any previous epoch
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

Call `reset()` before each evaluation pass. The evaluator accumulates across `update()`
calls by design, so without it a second epoch is scored on top of the first.

### Constructor Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `metrics` | `['endpoints', 'heatmap']` | Which metrics to compute. Unrecognized names are ignored, so a typo yields an evaluator that silently measures nothing. |
| `do_postprocess` | `True` | Apply Collinear Line Clipping before the heatmap metric. Does not affect `endpoints`. |
| `nms_thresh` | `0.01` | Clipping tolerance, as a fraction of the image diagonal. |
| `img_size` | `128.0` | Spatial resolution the normalized input coordinates are scaled to for the structural metrics. |

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
| `size`  | `[2]` | **Required for Heatmap:** original image size as `(height, width)`, e.g. `[512, 512]`. Falls back to `[128, 128]` when absent, which silently rescales the heatmap metrics — pass it whenever `'heatmap'` is enabled. |

*Note: Coordinates are automatically scaled and flipped geometrically within the library to match standard benchmarks. Endpoints are evaluated at the `img_size` scale (128x128 by default), while Heatmap metrics are evaluated at the original image resolution taken from `size`.*

### Reading Results Programmatically

`summarize()` prints a table, but it also leaves the numbers on each worker's `stats`
dict, which is what you want inside a training loop:

```python
evaluator.accumulate()
evaluator.summarize()

endpoint_stats = evaluator.evaluators['endpoints'].stats
# {'sAP_5': 58.4, 'sAP_10': 64.1, 'sAP_15': 66.9,
#  'sF_5': 62.0, 'sF_10': 67.3, 'sF_15': 69.5}

heatmap_stats = evaluator.evaluators['heatmap'].stats
# {'AP': 81.2, 'F': 79.6}

all_stats = endpoint_stats | heatmap_stats
```

Values are class-averaged floats on a 0-100 scale. The keys are only populated by
`summarize()`, so call it before reading them; the `sAP_*`/`sF_*` keys follow whatever
`thresholds` the endpoint worker was built with.

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
