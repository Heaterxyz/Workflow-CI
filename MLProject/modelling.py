"""
==============================================================
 modelling.py (versi CI / MLflow Project)
 Retraining model klasifikasi churn - dijalankan oleh:
     mlflow run MLProject --env-manager local
==============================================================

Perbedaan dengan versi eksperimen (folder Membangun_model):
- Menerima hyperparameter melalui argumen (diatur di file `MLProject`)
- Menyimpan salinan model ke folder lokal `model/` agar mudah
  di-upload sebagai artefak CI dan dipakai `mlflow build-docker`
"""

import argparse
import os

# MLflow 3.x: izinkan file store lokal ./mlruns (mode lama) agar kompatibel
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = "namadataset_preprocessing/telco_churn_preprocessing.csv"
EXPERIMENT_NAME = "Telco_Churn_CI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retraining model churn (CI)")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-dir", default="model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Jika MLFLOW_TRACKING_URI tidak diatur (mis. dijalankan langsung),
    # gunakan tracking lokal ./mlruns. Saat dijalankan via `mlflow run`,
    # env tersebut diatur oleh workflow agar CLI dan skrip memakai store yang sama.
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(args.data_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"Data latih: {X_train.shape} | Data uji: {X_test.shape}")

    params = {
        "model_type": "RandomForestClassifier",
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "n_train_rows": X_train.shape[0],
        "n_features": X_train.shape[1],
    }

    # Saat dijalankan via `mlflow run`, MLFLOW_RUN_ID sudah diatur oleh MLflow CLI:
    # lanjutkan run tersebut (bukan membuat run baru).
    run_name = None if os.getenv("MLFLOW_RUN_ID") else "ci-random-forest"
    with mlflow.start_run(run_name=run_name) as run:
        # Manual logging: parameter
        mlflow.log_params(params)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Manual logging: metrik
        mlflow.log_metrics(metrics)

        # Manual logging: model (signature + input example)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # Simpan salinan model ke folder lokal (dipakai CI untuk upload & build-docker)
        mlflow.sklearn.save_model(
            sk_model=model,
            path=args.model_dir,
            signature=signature,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print("\n=== Hasil retraining CI ===")
        for key, value in metrics.items():
            print(f"{key:<10}: {round(value, 4)}")
        print("\nRun ID       :", run.info.run_id)
        print("Model lokal  :", os.path.abspath(args.model_dir))


if __name__ == "__main__":
    main()
