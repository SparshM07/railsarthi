# Railsarthi — Quick Setup & Run Guide

## 1. What is Railsarthi?
Railsarthi is an ML-powered railway ETA prediction system for Indian Railways. It uses live train information and predicts how delay will evolve across upcoming stations using recursive delay forecasting.

---

## 2. Clone the Repository
```bash
git clone <repository-url>
cd railsarthi
```

---

## 3. Python Environment & Dependencies
Railsarthi requires **Python 3.10+** (tested on Python 3.12 and 3.13).

### Step A: Create and activate a virtual environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step B: Install dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Required Large Files & Storage Placement
Due to GitHub file size limits, large datasets and legacy model binaries are not stored directly in the Git repository.

The production model file (`backend/model/champion_model_scheduled_segment_v2.txt`) is already tracked and included in the repository.

If you need full historical training or research reproduction, place the corresponding files in their respective folders:

| File | Where to Place | Needed For |
|---|---|---|
| `champion_model.txt` | `backend/model/` | Legacy V1 benchmark (optional) |
| `ir_train.csv` | `backend/dataset/` | Retraining baseline models (optional) |
| `ir_test.csv` | `backend/dataset/` | Test dataset evaluation (optional) |
| `ir_sample_submission.csv` | `backend/dataset/` | Submission template (optional) |
| Weather Research Data & Models | `backend/research/weather/` | Reproducing Candidate C weather research (optional) |

*Note: You do NOT need the optional dataset files just to run the live dashboard and demo.*

---

## 5. Running the Application

### Start the FastAPI backend:
**On Windows (PowerShell):**
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
*Or, if your virtual environment is already activated:*
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Open the Web Dashboard:
Open your browser and navigate to:
```text
http://127.0.0.1:8000/app
```

---

## 6. Production vs. Research Models
- **Production Model (Active)**: Uses the frozen Champion V2 LightGBM model (`backend/model/champion_model_scheduled_segment_v2.txt`) with 13 operational train/segment features.
- **Weather Candidate C Model (Research Only)**: A completed R&D benchmark (+2.13% MAE improvement). It is kept purely in `backend/research/weather/` and is **NOT** used for live production inference.

---

## 7. Troubleshooting

- **Missing Model Error**: Ensure `champion_model_scheduled_segment_v2.txt` exists in `backend/model/`.
- **Missing Dataset Error**: If running training or historical analysis, ensure CSV files are placed in `backend/dataset/`.
- **Dependency / Import Error**: Re-run `pip install -r requirements.txt` inside your virtual environment.
- **Port 8000 Already in Use**: Another process is using port 8000. Either stop that process or start uvicorn on a different port:
  ```powershell
  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080
  ```
  Then open `http://127.0.0.1:8080/app`.
