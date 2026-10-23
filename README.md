# ToolSense

ToolSense is a machine-learning research prototype for estimating CNC milling cutting-tool flank wear from machining sensor signals and process conditions. It combines a Python API with a React dashboard for model evaluation, prediction-error analysis, and feature-importance visualization.

> **Important:** This is a research prototype. It is not certified for industrial safety decisions or autonomous machine control. The reported metrics are exploratory cross-validation estimates, not results from an independent external test set.

## 1. Project Overview

ToolSense investigates whether machining sensor measurements and engineered signal features can help estimate tool flank wear.

**Current model:** Extra Trees Regressor (`ExtraTreesRegressor`)

**Prediction target:** `VB`, measured in millimetres.

**Model inputs:** 50 engineered features:
- 36 sensor statistics: six statistics for each of six sensor channels.
- 12 signal-shape statistics: skewness and kurtosis for each channel.
- 2 machining conditions: depth of cut (`DOC`) and feed.

## 2. Dataset and Research Reference

The project uses the NASA/UC Berkeley Milling Wear Dataset.

- Dataset: https://data.nasa.gov/dataset/milling-wear
- Research paper: Tim von Hahn and Chris K. Mechefske, “Machine Learning in CNC Machining: Best Practices,” *Machines*, 10(12), 1233 (2022). https://www.mdpi.com/2075-1702/10/12/1233
- Associated code: https://github.com/tvhahn/tspipe

The raw dataset is expected at `data/raw/mill.mat`.

The working dataset contains 167 records, of which 146 have measured flank-wear labels. Missing wear labels must not be treated as zero.

## 3. Current Model Results

The candidate model uses Extra Trees with 200 estimators, maximum depth 12, minimum samples per leaf 1, and `max_features=1.0`.

| Metric | Result |
|---|---:|
| MAE | 0.125979 mm |
| RMSE | 0.188323 mm |
| R² | 0.473881 |
| High-wear MAE | 0.258426 mm |
| Model features | 50 |
| Measured records | 146 |

**Validation method:** 5-fold GroupKFold, grouped by machining case.

The fold results vary, with fold 3 showing notably higher errors. High-wear records also have higher MAE than the dataset overall. These results indicate that further validation and improvement are needed.

## 4. Technology Stack

### Frontend
- React
- JavaScript
- Vite
- Chart.js
- `react-chartjs-2`
- CSS

### Backend and machine learning
- Python
- Flask
- NumPy
- pandas
- SciPy
- scikit-learn
- joblib
- pytest

## 5. Project Structure

```text
ToolSense/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
│   ├── public/
│   │   └── reports/
│   └── src/
├── models/
│   └── toolsense_model_candidate.joblib
├── notebooks/
├── reports/
│   ├── candidate_model_metrics.json
│   ├── candidate_model_predictions.csv
│   └── candidate_feature_importance.csv
├── src/
├── tests/
├── requirements.txt
└── README.md
```

## 6. Setup on Windows

### Prerequisites
- Python compatible with the installed environment
- Node.js and npm
- The dataset, saved model, and candidate metrics files

### Install Python dependencies

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the virtual environment already exists, activate it and install the requirements.

### Install frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

### Start the Flask backend

Open a terminal at the project root:

```powershell
.\.venv\Scripts\python.exe .\app\app.py
```

The API should be available at `http://127.0.0.1:5000`.

### Start the frontend

Open a second terminal:

```powershell
cd frontend
npm run dev
```

Open the local URL printed by Vite. In the current development setup, this is usually `http://localhost:5173/`.

Keep both terminals running while using the dashboard.

## 7. API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Backend status and loaded-model information |
| `GET /api/metrics` | Candidate-model metrics and validation-fold results |
| `GET /api/sample` | Real dataset sample with the expected model features |
| `POST /api/predict` | Predicts flank wear from the 50 input features |

For a prediction request, provide all 50 feature names returned by `/api/metrics`. Values must be numeric and finite. Use `/api/sample` to inspect a complete example input.

## 8. Dashboard Features

- Overview of model performance and backend status.
- Sensor and feature importance analysis.
- Actual-versus-predicted and residual plots.
- Error analysis by wear range.
- Grouped cross-validation fold results.
- Wear prediction using a real dataset sample.
- Light and dark themes.

## 9. Testing

Run the Python tests from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests -v
```

The current test suite contains 16 passing tests covering API behavior, input validation, signal statistics, and shape-feature extraction.

To check the frontend production build:

```powershell
cd frontend
npm run build
```

## 10. Research Limitations

1. **Small dataset:** only 146 records have measured wear labels.
2. **Fold variability:** performance differs across machining cases.
3. **High-wear error:** the high-wear subset has greater MAE than the overall dataset.
4. **No independent test set:** grouped cross-validation estimates are not a substitute for external validation.
5. **Feature importance is not causation:** impurity-based importance is model-specific and may be affected by correlated features.
6. **No safety certification:** predictions must not be the sole basis for maintenance, operator safety, or machine-control decisions.

## 11. Reproducibility

- Keep training and inference feature extraction consistent.
- Preserve the exact feature names and order expected by the saved model.
- Do not replace missing wear labels with zero.
- Record model settings, validation methodology, fold results, and generated reports when conducting new experiments.
- Rerun tests after changing the backend or feature-engineering code.

## 12. Attribution

Follow the dataset, research-paper, and dependency licensing terms. Add a separate project license if you intend to distribute ToolSense.
