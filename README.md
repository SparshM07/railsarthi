# RailPulse AI | SIH Railway Delay & Real-Time ETA Intelligence Platform

Enterprise-grade machine learning platform and real-time interactive command center for Indian Railways train delay prediction, cascading station ETAs, and journey risk modeling.

---

## Key Highlights

- **Interactive Command Center Frontend**: Modern dark-mode glassmorphism dashboard with Leaflet geospatial tracking, pulsing train vectors, countdown timers, and cascading station timelines.
- **Resilient Hybrid Deployment Engine**: Automatically switches between live RailRadar tracking and high-fidelity local timetable simulation. Runs anywhere out of the box with zero required API keys.
- **Dual-Model ML Architecture**:
  1. **Live Next-Station Delay Model (`champion_model.txt`)**: 13-feature LightGBM booster integrating live GPS interpolation, route geometry, hierarchical segment statistics, and weather telemetry.
  2. **Journey-Level Delay Model (`journey_delay_model.txt`)**: 40-feature LightGBM regressor trained on 1.28M journeys (2018–2023) evaluated on unseen 2024 holdout data with zero data leakage.
- **Cascading ETA Engine**: Solves complex railway edge cases including multi-day midnight date-shifts, heavily delayed stale timetables, and multi-tier confidence scoring (`HIGH`, `MEDIUM`, `LOW`).

---

## 🚀 Quickstart & Local Run

1. **Clone & Setup Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Service**:
   ```bash
   uvicorn backend.main:app --reload
   ```

3. **Open Dashboard**:
   - **Interactive Web App**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000) or [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app) in your browser.
   - **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🐳 Docker & 1-Command Deployment

Run the entire application in a hardened, non-root Docker container:

```bash
docker compose up --build -d
```

Access the dashboard at `http://localhost:8000`.

---

## ☁️ 1-Click Cloud Deployment (Render / Railway / Fly.io)

This repository includes native deployment manifests:
- **Render**: Uses `render.yaml` (Free Tier compatible).
- **Railway / Heroku / Generic PaaS**: Uses `Procfile` and `Dockerfile`.

---

## 🧪 Test Suite

Run the full offline test suite:

```bash
python -m unittest discover -s tests -v
```

All 22 unit and integration tests run entirely offline with mock isolation and zero external dependencies.

---

## 📡 Core API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` or `/app` | Serves the interactive RailPulse AI Command Center dashboard. |
| `GET` | `/trains` | Catalog of popular express trains available for instant demo testing. |
| `GET` | `/health` | Liveness/readiness probe reporting active mode (`LIVE` vs `SIMULATION_FALLBACK`). |
| `GET` | `/metrics` | Operational metrics (cache hit/miss ratios, status codes). Protected when auth is enabled. |
| `POST` | `/predict` | Live next-station delay prediction, route GeoJSON geometry, and cascading upcoming ETAs. |
| `POST` | `/predict-journey` | Offline journey-level delay simulation across 40 infrastructure and environmental features. |
