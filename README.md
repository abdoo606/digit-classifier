# 26 · digits

Softmax regression with gradients written by hand.

## Run

```bash
python -m digits.softmax  (or import the model)
```

## Test

```bash
pytest projects/26-digit-classifier -q
```

## What it demonstrates

- Forward pass, cross-entropy and dW/db derived and implemented in NumPy
- Numerically stable softmax (max-shift) - no overflow at large logits
- Mini-batch SGD with learning-rate decay and a per-epoch history
- Confusion matrix to see which classes get confused

## Repository
- GitHub: https://github.com/{owner}/digit-classifier
- Python 3.11+ · tests: `pytest` · runtime dependencies: numpy
