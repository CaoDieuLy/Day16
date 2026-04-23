import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path("/opt/ml-benchmark")
RESULT_PATH = BASE_DIR / "benchmark_result.json"
REPORT_PATH = BASE_DIR / "benchmark_report.txt"
DATASET_PATHS = [
    BASE_DIR / "creditcard.csv",
    Path.home() / "ml-benchmark" / "creditcard.csv",
]


def load_dataset():
    for dataset_path in DATASET_PATHS:
        if dataset_path.exists():
            frame = pd.read_csv(dataset_path)
            labels = frame["Class"].to_numpy()
            features = frame.drop(columns=["Class"]).to_numpy()
            return {
                "dataset": "kaggle_creditcardfraud",
                "source": str(dataset_path),
                "features": features,
                "labels": labels,
            }

    dataset = load_breast_cancer()
    return {
        "dataset": "sklearn_breast_cancer",
        "source": "sklearn.datasets.load_breast_cancer",
        "features": dataset.data,
        "labels": dataset.target,
    }


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset()
    features = data["features"]
    labels = data["labels"]

    split_start = time.perf_counter()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    load_seconds = time.perf_counter() - split_start

    train_start = time.perf_counter()
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    batch_predict_seconds = time.perf_counter() - predict_start

    one_row_start = time.perf_counter()
    model.predict_proba(x_test[:1])
    one_row_latency_seconds = time.perf_counter() - one_row_start

    thousand_rows = x_test[: min(1000, len(x_test))]
    throughput_start = time.perf_counter()
    model.predict_proba(thousand_rows)
    throughput_seconds = time.perf_counter() - throughput_start

    result = {
        "dataset": data["dataset"],
        "source": data["source"],
        "row_count": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "load_time_seconds": round(load_seconds, 6),
        "training_time_seconds": round(training_seconds, 6),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or model.n_estimators),
        "auc_roc": round(float(roc_auc_score(y_test, probabilities)), 6),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "f1_score": round(float(f1_score(y_test, predictions)), 6),
        "precision": round(float(precision_score(y_test, predictions)), 6),
        "recall": round(float(recall_score(y_test, predictions)), 6),
        "inference_latency_1_row_ms": round(float(one_row_latency_seconds * 1000), 6),
        "inference_throughput_1000_rows_per_sec": round(
            float(len(thousand_rows) / throughput_seconds), 6
        ),
        "batch_predict_seconds": round(float(batch_predict_seconds), 6),
    }

    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    REPORT_PATH.write_text(
        "\n".join(
            [
                "CPU fallback was used because the GCP project could not obtain GPU quota.",
                f"Dataset: {result['dataset']} ({result['row_count']} rows, {result['feature_count']} features).",
                f"Training time: {result['training_time_seconds']} seconds.",
                f"AUC-ROC: {result['auc_roc']}.",
                f"Accuracy: {result['accuracy']}, F1-score: {result['f1_score']}.",
                f"Precision: {result['precision']}, Recall: {result['recall']}.",
                f"Single-row inference latency: {result['inference_latency_1_row_ms']} ms.",
                f"Throughput for up to 1000 rows: {result['inference_throughput_1000_rows_per_sec']} rows/sec.",
                "This CPU path still demonstrates Terraform provisioning, ML training, inference, and billing validation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Benchmark completed successfully.")
    print(json.dumps(result, indent=2))
    print(f"benchmark_result.json written to: {RESULT_PATH}")
    print(f"benchmark_report.txt written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
