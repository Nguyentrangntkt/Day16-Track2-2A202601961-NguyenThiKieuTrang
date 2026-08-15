"""Local LightGBM benchmark for the credit-card fraud dataset.

Run this script from the directory containing both this file and
``creditcard.csv``. It writes ``benchmark_result.json`` to that directory.
"""

import json
import time
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2
LATENCY_RUNS = 1_000
BATCH_SIZE = 1_000


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_path = script_dir / "creditcard.csv"
    result_path = script_dir / "benchmark_result.json"

    load_start = time.perf_counter()
    dataset = pd.read_csv(dataset_path)
    load_time_seconds = time.perf_counter() - load_start

    if "Class" not in dataset.columns:
        raise ValueError("Dataset must contain a 'Class' target column.")

    X = dataset.drop(columns=["Class"])
    y = dataset["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    training_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time_seconds = time.perf_counter() - training_start

    fraud_probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    # Warm up prediction code paths before timing inference.
    one_row = X_test.iloc[:1]
    model.predict_proba(one_row)

    latency_total_seconds = 0.0
    for _ in range(LATENCY_RUNS):
        latency_start = time.perf_counter()
        model.predict_proba(one_row)
        latency_total_seconds += time.perf_counter() - latency_start
    inference_latency_ms_one_row = (latency_total_seconds / LATENCY_RUNS) * 1_000

    batch = X_test.iloc[: min(BATCH_SIZE, len(X_test))]
    batch_start = time.perf_counter()
    model.predict_proba(batch)
    batch_prediction_seconds = time.perf_counter() - batch_start
    inference_throughput_rows_per_second = len(batch) / batch_prediction_seconds

    best_iteration = getattr(model, "best_iteration_", None)
    # LightGBM reports 0 when no early-stopping best iteration exists.
    if best_iteration is None or best_iteration <= 0:
        best_iteration = None
    else:
        best_iteration = int(best_iteration)

    model_parameters = {
        key: value
        for key, value in model.get_params().items()
        if key in {"objective", "n_estimators", "learning_rate", "num_leaves", "random_state", "n_jobs"}
    }
    result = {
        "cloud": "aws",
        "instance_type": "t3.small",
        "dataset_rows": int(len(dataset)),
        "load_time_seconds": load_time_seconds,
        "training_time_seconds": training_time_seconds,
        "auc_roc": roc_auc_score(y_test, fraud_probabilities),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "inference_latency_ms_one_row": inference_latency_ms_one_row,
        "inference_throughput_rows_per_second": inference_throughput_rows_per_second,
        "best_iteration": best_iteration,
        "model_parameters": model_parameters,
        "batch_prediction_rows": int(len(batch)),
        "batch_prediction_time_seconds": batch_prediction_seconds,
    }

    with result_path.open("w", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2, allow_nan=False)
        result_file.write("\n")

    print("LightGBM benchmark completed")
    print(f"Dataset rows: {result['dataset_rows']}")
    print(f"Load time: {result['load_time_seconds']:.4f} seconds")
    print(f"Training time: {result['training_time_seconds']:.4f} seconds")
    print(f"AUC-ROC: {result['auc_roc']:.6f}")
    print(f"Accuracy: {result['accuracy']:.6f}")
    print(f"Precision: {result['precision']:.6f}")
    print(f"Recall: {result['recall']:.6f}")
    print(f"F1 score: {result['f1_score']:.6f}")
    print(f"One-row inference latency: {result['inference_latency_ms_one_row']:.4f} ms")
    print(f"Inference throughput: {result['inference_throughput_rows_per_second']:.2f} rows/second")
    print(f"Best iteration: {result['best_iteration']}")
    print(f"Results written to: {result_path}")


if __name__ == "__main__":
    main()
