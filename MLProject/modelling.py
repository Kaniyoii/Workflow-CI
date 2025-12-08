# Library
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

def parse_max_depth(x):
    if x is None:
        return None
    x = str(x).strip().lower()
    if x in ["none", "null", ""]:
        return None
    return int(x)

# Modeling
def main(args):
    # Load dataset hasil preprocessing
    train_df = pd.read_csv(args.train_path)
    test_df  = pd.read_csv(args.test_path)

    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]

    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"]

    # MLflow experiment (untuk CI)
    mlflow.set_experiment("obesity-ci")

    # autolog
    mlflow.sklearn.autolog()

    # Training model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)

    # Evaluasi di test set
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # log test metrik manual
    mlflow.log_metrics({
        "test_accuracy": acc,
        "test_precision_weighted": prec,
        "test_recall_weighted": rec,
        "test_f1_weighted": f1
    })

    # simpan model ke workspace CI
    workspace = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
    save_path = os.path.join(workspace, "exported_model")
    mlflow.sklearn.save_model(model, save_path)
    print("Saved model to:", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="obesity_preprocessing/train.csv")
    parser.add_argument("--test_path", type=str, default="obesity_preprocessing/test.csv")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=parse_max_depth, default=None)  # <---
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()
    main(args)