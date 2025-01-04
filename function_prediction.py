import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge


def get_predictor(train_X, train_y, val_X, val_y):
    """
    Train a predictor using training data and evaluate it using validation data.

    Args:
        train_X (np.ndarray): Training features with shape (n_samples, n_features).
        train_y (np.ndarray): Training labels.
        val_X (np.ndarray): Validation features with shape (n_samples, n_features).
        val_y (np.ndarray): Validation labels.
    Returns:
        reg: Trained regressor.
        MSE: Mean squared error on validation data.
    """
    reg = KernelRidge(kernel="rbf", alpha=1.0).fit(train_X, train_y)
    val_pred = reg.predict(val_X)
    MSE = np.mean((val_pred - val_y) ** 2)
    return reg, MSE


if __name__ == "__main__":
    import pandas as pd
    from embedding import get_embedding
    import matplotlib.pyplot as plt

    train_df = pd.read_json(
        "data/fluorescence/fluorescence_train.json", lines=False
    ).rename(columns={"primary": "sequence"})
    train_df["log_fluorescence"] = train_df["log_fluorescence"].map(lambda x: x[0])
    val_df = pd.read_json(
        "data/fluorescence/fluorescence_valid.json", lines=False
    ).rename(columns={"primary": "sequence"})
    val_df["log_fluorescence"] = val_df["log_fluorescence"].map(lambda x: x[0])
    test_df = pd.read_json(
        "data/fluorescence/fluorescence_test.json", lines=False
    ).rename(columns={"primary": "sequence"})
    test_df["log_fluorescence"] = test_df["log_fluorescence"].map(lambda x: x[0])

    model_name = "esm2_t6_8M_UR50D"
    all_X = get_embedding(
        model_name,
        train_df["sequence"].tolist()
        + val_df["sequence"].tolist()
        + test_df["sequence"].tolist(),
        batch_size=16,
    )
    all_X = (all_X - all_X.mean(axis=0)) / all_X.std(axis=0)
    train_X = all_X[: len(train_df)].copy()
    val_X = all_X[len(train_df) : len(train_df) + len(val_df)].copy()
    test_X = all_X[len(train_df) + len(val_df) :].copy()
    train_y = train_df["log_fluorescence"].values
    val_y = val_df["log_fluorescence"].values
    test_y = test_df["log_fluorescence"].values

    reg, val_MSE = get_predictor(train_X, train_y, val_X, val_y)
    test_pred = reg.predict(test_X)
    test_MSE = np.mean((test_pred - test_y) ** 2)
    print(f"Validation MSE: {val_MSE:.4f}")
    print(f"Test MSE: {test_MSE:.4f}")

    plt.figure(figsize=(5, 5))
    plt.scatter(test_y, test_pred, alpha=0.5, marker=".")
    plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], "r--")
    plt.xlabel("True log fluorescence")
    plt.ylabel("Predicted log fluorescence")
    plt.savefig("test_prediction.png", dpi=300, bbox_inches="tight")
