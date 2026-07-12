# Heart Disease Predictions MLOps

End-to-end MLOps project for predicting heart disease risk using the UCI Heart Disease dataset.

## Assignment Coverage

- Data acquisition and EDA
- Feature preprocessing and reproducible model training
- Logistic Regression and Random Forest model comparison
- MLflow experiment tracking
- Saved model and preprocessing artifacts
- Flask prediction API with Prometheus metrics
- Unit tests with Pytest
- GitHub Actions CI workflow
- Dockerfile for containerized serving
- Kubernetes deployment and service manifests
- Report and screenshot scaffolding

## Project Structure

```text
src/
  components/
    data_ingestion.py
    model_transformation.py
    model_trainer.py
  pipeline/
    train_pipeline.py
    predict_pipeline.py
app.py
notebooks/
  EDA.ipynb
  Model_Training.ipynb
tests/
.github/workflows/ci.yml
k8s/
monitoring/
docs/
```

## Local Setup

You already have a local Conda environment in `.conda`. On Windows PowerShell, run commands with:

```powershell
.\.conda\python.exe -m pip install -r requirements.txt
.\.conda\python.exe -m pip install -e .
```

If creating a fresh environment:

```bash
conda env create -f environment.yml
conda activate heart-disease-mlops
```

## Train The Model

```powershell
.\.conda\python.exe -m src.pipeline.train_pipeline
```

This creates:

- `artifacts/data/raw.csv`
- `artifacts/data/train.csv`
- `artifacts/data/test.csv`
- `artifacts/preprocessor.pkl`
- `artifacts/model.pkl`
- `artifacts/plots/model_comparison.png`
- MLflow runs under `mlruns/`

## Run A Sample Prediction

```powershell
.\.conda\python.exe scripts\predict_sample.py
```

## Run The API

```powershell
.\.conda\python.exe app.py
```

The application will automatically open in your default web browser at `http://127.0.0.1:5000/`. You can use the modern, AJAX-powered web interface to enter patient data and receive instant predictions.

Endpoints:

- `GET /health`
- `POST /predict`
- `GET /metrics`

PowerShell sample:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -InFile sample_request.json
```

## MLflow

```powershell
.\.conda\Scripts\mlflow.exe ui
```

Open the URL printed by MLflow and capture screenshots for the report.

## Tests

```powershell
.\.conda\python.exe -m pytest -q tests
```

## Docker

Train the model first so `artifacts/model.pkl` and `artifacts/preprocessor.pkl` exist, then build:

```bash
docker build -t heart-disease-api:latest .
docker run -p 5000:5000 heart-disease-api:latest
```

## Kubernetes

For Docker Desktop or Minikube:

```bash
kubectl apply -f k8s/
kubectl get pods
kubectl get svc
```

If using Minikube, make sure the image is available inside Minikube before applying manifests.

## Monitoring

The API exposes Prometheus metrics at `/metrics`.

Prometheus config is available at:

```text
monitoring/prometheus.yml
```

## Documentation & Report

The final MLOps report has been compiled and is available in:
- Markdown format: `docs/report.md`
- PDF format: `docs/report.pdf` (Generated via `scripts/md_to_pdf.py`)

All required screenshots (EDA, MLflow, CI/CD, Docker, Kubernetes) are categorized in `docs/screenshots/`.

## Manual Items Still Required

- **Video Walkthrough**: Record the short video walkthrough demonstrating the overall pipeline as requested by the assignment.
