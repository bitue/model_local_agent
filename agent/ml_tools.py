"""
ml_tools.py
Machine Learning tool functions exposed to the autonomous ReAct agent.

Each tool:
  - accepts simple, JSON-serializable keyword arguments (so the LLM's
    ``Action Input: {...}`` maps directly onto **kwargs)
  - returns a JSON string (so it can be read back as an Observation)
  - never raises on *expected* bad input (unknown dataset/model/param) --
    it returns {"error": "..."} instead, with an actionable message so the
    agent can self-correct (Task 3)
  - unexpected exceptions (e.g. a genuine shape mismatch) are allowed to
    propagate; react_agent.py catches them and turns them into an
    Observation too
"""

import json
import math

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim


DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def _get_dataset(name: str):
    """Resolves and loads a dataset by name, or raises ValueError."""
    name = name.lower().strip()
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(DATASETS.keys())}")
    return name, DATASETS[name]()


# ---------------------------------------------------------------------------
# Task 1 -- baseline tools
# ---------------------------------------------------------------------------

def load_dataset_summary(dataset_name: str) -> str:
    """Loads a standard benchmark dataset and returns summary statistics."""
    try:
        name, data = _get_dataset(dataset_name)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    summary = {
        "dataset": name,
        "n_samples": df.shape[0],
        "n_features": len(data.feature_names),
        "feature_names": list(data.feature_names),
        "classes": [str(c) for c in np.unique(data.target)],
        "missing_values": int(df.isnull().sum().sum()),
    }
    return json.dumps(summary)


def train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2) -> str:
    """Trains a Scikit-Learn model (decision_tree, logistic_regression, random_forest)."""
    try:
        name, data = _get_dataset(dataset_name)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    if not 0.0 < test_size < 1.0:
        return json.dumps({"error": f"test_size must be between 0 and 1, got {test_size}."})

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=42, stratify=data.target
    )

    model_type = model_type.lower().strip()
    if model_type == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    elif model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        return json.dumps({
            "error": f"Unsupported model '{model_type}'. "
                     f"Valid options: decision_tree, logistic_regression, random_forest."
        })

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(clf, data.data, data.target, cv=5)

    return json.dumps({
        "model": model_type,
        "dataset": name,
        "test_accuracy": round(acc, 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
    })


def train_pytorch_mlp(dataset_name: str, hidden_dim: int = 32, epochs: int = 50, lr: float = 0.01) -> str:
    """Trains a simple PyTorch MLP (Linear-ReLU-Linear) on the selected dataset."""
    try:
        name, data = _get_dataset(dataset_name)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    if hidden_dim <= 0:
        return json.dumps({"error": f"hidden_dim must be a positive integer, got {hidden_dim}."})
    if epochs <= 0:
        return json.dumps({"error": f"epochs must be a positive integer, got {epochs}."})

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    num_features = X_train.shape[1]
    num_classes = len(np.unique(data.target))

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.long)

    model = nn.Sequential(
        nn.Linear(num_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    final_loss = None
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

        if math.isnan(final_loss):
            return json.dumps({
                "error": "Training diverged: loss became NaN. "
                         "Retry with a smaller learning rate (e.g. lr=0.001) or fewer epochs.",
                "framework": "PyTorch",
                "dataset": name,
                "hidden_dim": hidden_dim,
                "lr": lr,
            })

    with torch.no_grad():
        test_out = model(X_val_t)
        test_preds = torch.argmax(test_out, dim=1)
        acc = (test_preds == y_val_t).float().mean().item()

    return json.dumps({
        "framework": "PyTorch",
        "dataset": name,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "final_loss": round(float(final_loss), 4),
        "test_accuracy": round(acc, 4),
    })


# ---------------------------------------------------------------------------
# Task 2 -- advanced tools
# ---------------------------------------------------------------------------

_TUNE_PARAM_GRIDS = {
    "svc": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
    "decision_tree": {
        "max_depth": [2, 4, 6, 8, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    },
}


def tune_hyperparameters(dataset_name: str, model_type: str, search_type: str = "grid", cv: int = 5) -> str:
    """
    Hyperparameter-tunes an SVC (Kernel SVM) or DecisionTreeClassifier using
    GridSearchCV or RandomizedSearchCV. Returns the best parameters found and
    the best cross-validated score.
    """
    try:
        name, data = _get_dataset(dataset_name)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    model_type = model_type.lower().strip()
    if model_type not in _TUNE_PARAM_GRIDS:
        return json.dumps({
            "error": f"Unsupported model '{model_type}' for tuning. "
                     f"Valid options: {list(_TUNE_PARAM_GRIDS.keys())}."
        })

    base_model = SVC(random_state=42) if model_type == "svc" else DecisionTreeClassifier(random_state=42)
    param_grid = _TUNE_PARAM_GRIDS[model_type]

    search_type = search_type.lower().strip()
    if search_type == "grid":
        search = GridSearchCV(base_model, param_grid, cv=cv, n_jobs=-1)
    elif search_type == "random":
        search = RandomizedSearchCV(base_model, param_grid, cv=cv, n_jobs=-1, random_state=42, n_iter=10)
    else:
        return json.dumps({"error": f"Unsupported search_type '{search_type}'. Use 'grid' or 'random'."})

    search.fit(data.data, data.target)

    return json.dumps({
        "model": model_type,
        "search_type": search_type,
        "dataset": name,
        "best_params": search.best_params_,
        "best_cv_score": round(float(search.best_score_), 4),
    })


def reduce_dimensionality(dataset_name: str, method: str = "pca", n_components: int = 2) -> str:
    """
    Applies dimensionality reduction / feature selection:
      - method='pca': Principal Component Analysis, returns explained variance ratio.
      - method='sequential': Sequential (forward) Feature Selection, returns selected feature names.
    """
    try:
        name, data = _get_dataset(dataset_name)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    method = method.lower().strip()
    max_features = data.data.shape[1]
    if n_components <= 0 or n_components > max_features:
        return json.dumps({
            "error": f"n_components must be between 1 and {max_features} for dataset '{name}', "
                     f"got {n_components}."
        })

    if method == "pca":
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(data.data)
        return json.dumps({
            "method": "pca",
            "dataset": name,
            "n_components": n_components,
            "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "total_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
        })

    if method == "sequential":
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        selector = SequentialFeatureSelector(estimator, n_features_to_select=n_components, direction="forward")
        selector.fit(data.data, data.target)
        selected = [f for f, keep in zip(data.feature_names, selector.get_support()) if keep]
        return json.dumps({
            "method": "sequential",
            "dataset": name,
            "n_components": n_components,
            "selected_features": selected,
        })

    return json.dumps({"error": f"Unsupported method '{method}'. Use 'pca' or 'sequential'."})


def train_deep_classifier(
    dataset_name: str,
    hidden_dims=None,
    dropout: float = 0.3,
    use_batchnorm: bool = True,
    epochs: int = 50,
    lr: float = 0.01,
    scheduler: str = "steplr",
) -> str:
    """
    Trains a configurable, regularized PyTorch classifier with Dropout,
    BatchNorm1d, and a learning-rate scheduler.
    hidden_dims: list of hidden layer sizes, e.g. [64, 32].
    scheduler: 'steplr' or 'plateau'.
    """
    try:
        name, data = _get_dataset(dataset_name)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    if hidden_dims is None:
        hidden_dims = [64, 32]
    if not isinstance(hidden_dims, list) or not all(isinstance(h, int) and h > 0 for h in hidden_dims):
        return json.dumps({"error": f"hidden_dims must be a list of positive integers, got {hidden_dims}."})
    if not 0.0 <= dropout < 1.0:
        return json.dumps({"error": f"dropout must be in [0, 1), got {dropout}."})

    scheduler = scheduler.lower().strip()
    if scheduler not in {"steplr", "plateau"}:
        return json.dumps({"error": f"Unsupported scheduler '{scheduler}'. Use 'steplr' or 'plateau'."})

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    num_features = X_train.shape[1]
    num_classes = len(np.unique(data.target))

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.long)

    layers = []
    in_dim = num_features
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, num_classes))
    model = nn.Sequential(*layers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if scheduler == "steplr":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 5, 1), gamma=0.5)
    else:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    final_loss = None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

        if math.isnan(final_loss):
            return json.dumps({
                "error": "Training diverged: loss became NaN. "
                         "Retry with a smaller learning rate (e.g. lr=0.001).",
                "framework": "PyTorch-DeepClassifier",
                "dataset": name,
                "hidden_dims": hidden_dims,
                "lr": lr,
            })

        if scheduler == "steplr":
            lr_scheduler.step()
        else:
            lr_scheduler.step(final_loss)

    model.eval()
    with torch.no_grad():
        test_out = model(X_val_t)
        test_preds = torch.argmax(test_out, dim=1)
        acc = (test_preds == y_val_t).float().mean().item()

    return json.dumps({
        "framework": "PyTorch-DeepClassifier",
        "dataset": name,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "use_batchnorm": use_batchnorm,
        "scheduler": scheduler,
        "epochs": epochs,
        "final_loss": round(float(final_loss), 4),
        "test_accuracy": round(acc, 4),
    })


# ---------------------------------------------------------------------------
# Tool registry -- maps the string name the LLM emits in "Action:" to the
# actual Python callable.
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "train_pytorch_mlp": train_pytorch_mlp,
    "tune_hyperparameters": tune_hyperparameters,
    "reduce_dimensionality": reduce_dimensionality,
    "train_deep_classifier": train_deep_classifier,
}
