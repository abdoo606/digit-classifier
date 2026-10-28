from __future__ import annotations

import numpy as np
import pytest

from digits.softmax import (
    SIZE,
    SoftmaxModel,
    cross_entropy,
    make_dataset,
    make_templates,
    softmax,
)


def test_templates_are_distinct():
    templates = make_templates()
    assert len(templates) == 5
    assert all(t.shape == (SIZE, SIZE) for t in templates.values())
    assert not np.allclose(templates[0], templates[1])


def test_dataset_shapes_and_balance():
    X, y = make_dataset(samples_per_class=10, seed=1)
    assert X.shape == (50, SIZE * SIZE)
    assert set(np.unique(y)) == {0, 1, 2, 3, 4}
    assert X.min() >= 0 and X.max() <= 1


def test_dataset_is_deterministic_for_a_seed():
    a = make_dataset(5, seed=3)[0]
    b = make_dataset(5, seed=3)[0]
    assert np.allclose(a, b)


def test_softmax_rows_sum_to_one():
    probs = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(probs > 0)


def test_softmax_is_numerically_stable():
    probs = softmax(np.array([[1000.0, 1000.0, 1000.0]]))
    assert not np.isnan(probs).any()
    assert probs.sum() == pytest.approx(1.0)


def test_cross_entropy_perfect_prediction_is_zero():
    probs = np.array([[0.0, 1.0]])
    assert cross_entropy(probs, np.array([1])) == pytest.approx(0.0, abs=1e-6)


def test_cross_entropy_penalises_confident_mistakes():
    wrong = cross_entropy(np.array([[0.99, 0.01]]), np.array([1]))
    right = cross_entropy(np.array([[0.01, 0.99]]), np.array([1]))
    assert wrong > right


def test_model_learns_above_chance():
    X, y = make_dataset(samples_per_class=25, seed=2)
    model = SoftmaxModel(n_features=SIZE * SIZE, n_classes=5, epochs=25, learning_rate=0.6)
    model.fit(X, y)
    assert model.accuracy(X, y) > 0.7  # chance would be 0.2


def test_training_loss_decreases():
    X, y = make_dataset(samples_per_class=15, seed=4)
    model = SoftmaxModel(n_features=SIZE * SIZE, n_classes=5, epochs=15).fit(X, y)
    assert model.history[-1]["loss"] < model.history[0]["loss"]


def test_gradients_shrink_the_loss_on_one_batch():
    X, y = make_dataset(samples_per_class=4, seed=5)
    model = SoftmaxModel(n_features=SIZE * SIZE, n_classes=5, epochs=1, batch_size=len(y), learning_rate=0.5)
    before = cross_entropy(model.predict_proba(X), y)
    model.fit(X, y)
    after = cross_entropy(model.predict_proba(X), y)
    assert after < before


def test_confusion_matrix_is_square_and_counts_everything():
    X, y = make_dataset(samples_per_class=5, seed=6)
    model = SoftmaxModel(n_features=SIZE * SIZE, n_classes=5, epochs=1).fit(X, y)
    matrix = model.confusion_matrix(X, y)
    assert matrix.shape == (5, 5)
    assert matrix.sum() == len(y)


def test_history_records_every_epoch():
    model = SoftmaxModel(n_features=SIZE * SIZE, n_classes=5, epochs=3)
    assert len(model.history) == 0
    X, y = make_dataset(samples_per_class=2, seed=7)
    model.fit(X, y)
    assert len(model.history) == 3
    assert model.history[0]["epoch"] == 0
