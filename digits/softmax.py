"""Multinomial logistic regression: forward pass, cross-entropy, analytic gradients.

The dataset is generated from templates so the project needs no downloads, yet still
exercises the full training loop: shuffling, mini-batches, learning-rate decay,
accuracy tracking and a confusion matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIZE = 8


def make_templates() -> dict[int, np.ndarray]:
    """Crude 8x8 templates for the digits 0-4 - enough structure to be learnable."""
    def blank() -> np.ndarray:
        return np.zeros((SIZE, SIZE))

    templates = {}
    zero = blank(); zero[1:7, 2] = 1; zero[1:7, 5] = 1; zero[1, 3:5] = 1; zero[6, 3:5] = 1
    one = blank(); one[1:7, 3] = 1; one[6, 2:5] = 1; one[1, 2:4] = 1
    two = blank(); two[1, 2:6] = 1; two[2:4, 5] = 1; two[4, 2:6] = 1; two[5:6, 2] = 1; two[6, 2:6] = 1
    three = blank(); three[1, 2:6] = 1; three[2:3, 5] = 1; three[4, 2:6] = 1; three[5, 5] = 1; three[6, 2:6] = 1
    four = blank(); four[1:4, 2] = 1; four[1:4, 5] = 1; four[4, 1:7] = 1; four[5:6, 5] = 1
    for label, template in enumerate((zero, one, two, three, four)):
        templates[label] = template
    return templates


def make_dataset(
    samples_per_class: int = 60, *, noise: float = 0.12, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    templates = make_templates()
    X, y = [], []
    for label, template in templates.items():
        for _ in range(samples_per_class):
            shifted = np.roll(template, rng.integers(-1, 2), axis=0)
            shifted = np.roll(shifted, rng.integers(-1, 2), axis=1)
            flipped = rng.random((SIZE, SIZE)) < noise
            noisy = np.clip(shifted * 0.85 + 0.15 + rng.normal(0, 0.08, (SIZE, SIZE)), 0, 1)
            noisy[flipped] = rng.random(flipped.sum())
            X.append(noisy.ravel())
            y.append(label)
    return np.array(X), np.array(y)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Subtracts the row max first: exp(1000) overflows, exp(0) does not."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    correct = probs[np.arange(n), y]
    return float(-np.mean(np.log(np.clip(correct, 1e-12, 1.0))))


@dataclass(slots=True)
class SoftmaxModel:
    n_features: int
    n_classes: int
    learning_rate: float = 0.5
    epochs: int = 40
    batch_size: int = 32
    l2: float = 1e-3
    weights: np.ndarray = field(init=False)
    bias: np.ndarray = field(init=False)
    history: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(42)
        self.weights = rng.normal(0, 0.01, (self.n_features, self.n_classes))
        self.bias = np.zeros(self.n_classes)

    def logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.logits(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxModel":
        rng = np.random.default_rng(0)
        n = len(y)
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                probs = self.predict_proba(X[batch])
                # d(loss)/d(logits) for softmax + cross-entropy collapses to (p - onehot)
                error = probs
                error[np.arange(len(batch)), y[batch]] -= 1
                error /= len(batch)
                grad_w = X[batch].T @ error + self.l2 * self.weights
                grad_b = error.sum(axis=0)
                lr = self.learning_rate / (1 + 0.05 * epoch)  # simple decay
                self.weights -= lr * grad_w
                self.bias -= lr * grad_b
            self.history.append(
                {"epoch": float(epoch), "loss": round(cross_entropy(self.predict_proba(X), y), 4),
                 "accuracy": round(self.accuracy(X, y), 4)}
            )
        return self

    def confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        predicted = self.predict(X)
        matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for actual, guess in zip(y, predicted):
            matrix[actual, guess] += 1
        return matrix
