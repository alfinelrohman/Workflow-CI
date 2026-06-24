import os
import argparse
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="churn_preprocessing/dataset_ready.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.data_path)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=2,
        random_state=args.random_state,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("min_samples_split", 2)
    mlflow.log_param("min_samples_leaf", 2)
    mlflow.log_param("test_size", args.test_size)
    mlflow.log_param("random_state", args.random_state)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    os.makedirs("artifacts", exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - CI")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = "artifacts/confusion_matrix_ci.png"
    plt.savefig(cm_path)
    plt.close()

    report_path = "artifacts/classification_report_ci.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, y_pred))

    model_path = "artifacts/model.joblib"
    joblib.dump(model, model_path)

    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(report_path)
    mlflow.log_artifact(model_path)
    mlflow.sklearn.log_model(model, "model")

    print("CI training selesai")
    print(f"accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")


if __name__ == "__main__":
    main()