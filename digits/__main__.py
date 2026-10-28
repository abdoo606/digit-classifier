"""CLI: train the model and print the learning curve."""
from .softmax import SIZE, SoftmaxModel, make_dataset

if __name__ == "__main__":
    X, y = make_dataset(samples_per_class=40, seed=0)
    model = SoftmaxModel(n_features=SIZE * SIZE, n_classes=5, epochs=30).fit(X, y)
    for row in model.history[::5]:
        print(f"epoch {int(row['epoch']):>3}  loss={row['loss']:.4f}  acc={row['accuracy']:.3f}")
    print("final accuracy:", model.accuracy(X, y))
