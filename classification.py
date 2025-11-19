import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def load_data(data_path: Path, cols_path: Path):
    columns = [line.strip() for line in cols_path.read_text().splitlines() if line.strip()]
    df = pd.read_csv(
        data_path,
        header=None,
        names=columns,
        na_values=["?", " ?"],
        skipinitialspace=True,
    )
    raw_labels = df["label"].astype(str).str.strip()

    def parse_label(val):
        v = str(val).replace(" ", "").replace(".", "")
        if v.startswith("-"):
            return 0
        elif v.startswith("50000+"):
            return 1
        else:
            raise ValueError(f"Unexpected label format: {val}")

    y = raw_labels.apply(parse_label)

    if "weight" in df.columns:
        X = df.drop(columns=["label", "weight"])
    elif "fnlwgt" in df.columns:
        X = df.drop(columns=["label", "fnlwgt"])
    else:
        X = df.drop(columns=["label"])
    return X, y


def preprocess_features(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    return X_processed


def evaluate_sklearn_model(name, model, X_train, X_test, y_train, y_test, results):
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-12)
    else:
        y_proba = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    results.append(
        {
            "model_name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
	}
        )


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, X_test, y_train, y_test, results, device):
    X_train_t = torch.from_numpy(X_train.astype("float32"))
    X_test_t = torch.from_numpy(X_test.astype("float32"))
    y_train_t = torch.from_numpy(y_train.values.astype("float32")).unsqueeze(1)
    y_test_t = torch.from_numpy(y_test.values.astype("float32")).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)

    model = MLP(X_train.shape[1]).to(device)
    pos_frac = y_train.mean()
    neg_frac = 1 - pos_frac
    pos_weight_value = neg_frac / pos_frac
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    n_epochs = 10
    print("\n===== MLP training =====")
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch:02d}/{n_epochs} - Train BCE loss: {avg_loss:.4f}")
    print("Training finished")

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device)).cpu().numpy().reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, probs)

    print("\n===== MLP (3-layer, pos_weight, thresh=0.5) =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    results.append(
        {
            "model_name": "MLP (3-layer, pos_weight)",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
	}
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="census-bureau.data")
    parser.add_argument("--columns", type=str, default="census-bureau.columns")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    X, y = load_data(Path(args.data), Path(args.columns))
    X_processed = preprocess_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    model_results = []

    log_reg = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs",
    )
    evaluate_sklearn_model(
        "LogisticRegression (L2, balanced)",
        log_reg,
        X_train,
        X_test,
        y_train,
        y_test,
        model_results,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=10,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=args.random_state,
    )
    evaluate_sklearn_model(
        "RandomForest (300 trees, balanced_subsample)",
        rf,
        X_train,
        X_test,
        y_train,
        y_test,
        model_results,
    )

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=args.random_state,
    )
    evaluate_sklearn_model(
        "HistGradientBoosting (leaf=31, lr=0.1)",
        hgb,
        X_train,
        X_test,
        y_train,
        y_test,
        model_results,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mlp(X_train, X_test, y_train, y_test, model_results, device)

    df_results = pd.DataFrame(model_results).set_index("model_name")
    print("\n=== Model comparison (sorted by ROC-AUC) ===")
    print(df_results.sort_values("roc_auc", ascending=False))


if __name__ == "__main__":
    main()
