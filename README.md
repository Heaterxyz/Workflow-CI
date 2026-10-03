# Workflow-CI — Retraining Model Otomatis (MLflow Project)

Repository CI untuk melakukan **retraining model** klasifikasi Telco Customer Churn secara otomatis menggunakan **MLflow Project** + **GitHub Actions**.

## Struktur

```
Workflow-CI/
├── .github/workflows/ci.yml                  # Workflow CI (trigger: push & manual)
├── MLProject/
│   ├── MLProject                             # Deskriptor MLflow Project (entry point `main`)
│   ├── conda.yaml                            # Environment dependencies
│   ├── modelling.py                          # Script retraining (manual logging MLflow)
│   └── namadataset_preprocessing/
│       └── telco_churn_preprocessing.csv     # Dataset siap latih (hasil Kriteria 1)
├── artifacts/                                # Artefak model hasil CI (di-commit otomatis)
├── requirements.txt
└── README.md
```

## Tautan Docker Hub

**Docker Image (Advanced):**
`https://hub.docker.com/r/<DOCKERHUB_USERNAME>/telco-churn-model`

Cara menjalankan image (setelah CI push):
```bash
docker pull <DOCKERHUB_USERNAME>/telco-churn-model:latest
docker run -p 5001:8080 <DOCKERHUB_USERNAME>/telco-churn-model:latest
# endpoint: http://localhost:5001/invocations
```

## Menjalankan MLflow Project secara Lokal

```bash
pip install -r requirements.txt
mlflow run MLProject --env-manager local --experiment-name Telco_Churn_CI
```

Hasil: model tersimpan di `MLProject/model/`, tracking run di `mlruns/`.

## Yang Dilakukan Workflow CI

1. **Basic** — retraining model dijalankan melalui `mlflow run MLProject` setiap trigger (push / manual).
2. **Skilled** — artefak model disimpan ke repository GitHub yang sama (folder `artifacts/`) dan di-upload sebagai GitHub Actions artifact.
3. **Advanced** — image Docker dibuat dengan `mlflow models build-docker` (pada MLflow 3.x, bentuk baru dari `mlflow build-docker`) lalu di-push ke Docker Hub. Membutuhkan secrets:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN` (access token Docker Hub)
