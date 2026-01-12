# Line Segment Evaluation

A high-performance, C++ optimized library for evaluating line segment detection models. This library serves as a drop-in evaluation engine that computes **Structural Average Precision (sAP)** and **F1 Scores** significantly faster than pure Python implementations.

Designed with a modular architecture similar to `pycocotools`, it supports batch processing, incremental updates, and global accumulation of metrics.

## ✅ TODO

The following features are currently in development or planned:

- [ ] **Heatmap Metrics (APh / Fh):** Pixel-level evaluation using C++ rasterization (Bresenham's algorithm). Currently disabled due to mismatching with DT-LSD reported results.


## 🚀 Features

* **C++ Backend:** Core matching logic (`Greedy Match`) and distance calculations are implemented in C++14 using `pybind11` for maximum speed.
* **Metric Support:**
    * **sAP (Structural Average Precision):** Calculated at thresholds 5, 10, and 15.
    * **F1 Score:** Reports the maximum possible F1 score across all confidence thresholds.
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
evaluator = LineEvaluator(metrics=['lines'])

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

**Ground Truths (`list` of `dict`):**
A list where each item corresponds to one image in the batch.

| Key | Shape | Description |
| :--- | :--- | :--- |
| `lines` | `[M, 2, 2]` or `[M, 4]` | Ground truth segments |

*Note: Coordinates are automatically scaled by 128.0 and flipped geometrically within the library to match standard benchmarks.*

## 📊 Metrics Explained

### sAP (Structural Average Precision)
The library computes the Area Under the Precision-Recall Curve (AUC) at specific distance thresholds.
* **sAP5:** Strict match (Distance < 5)
* **sAP10:** Standard match (Distance < 10)
* **sAP15:** Loose match (Distance < 15)

### F1 Score
Unlike sAP which integrates over all recall levels, the F1 score reported is the **Maximum F1**:

$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

It finds the optimal confidence threshold for your model that maximizes this score.
