# RailPulse AI

RailPulse AI is a FastAPI application for Indian Railways delay prediction and real-time ETA intelligence. It provides a browser-based command center, a live next-station prediction API, and a journey-level delay simulator backed by LightGBM models.

The service works without credentials: when a RailRadar key is not configured (or a provider request fails), it uses the built-in timetable/GPS simulator. With a valid key, live train and route data are fetched from RailRadar. Current weather enrichment is retrieved from Open-Meteo when available.

## What the application does

- Displays a dark-mode dashboard with a Leaflet route map, train position, delay status, station timeline, ETA countdown, and 15-second refresh.
- Predicts the next-station delay using train/station categories, current delay, scheduled segment duration, historical segment statistics, calendar features, and previous-station delay.
- Cascades the prediction into upcoming station ETAs, including overnight date rollovers and confidence levels (`HIGH`, `MEDIUM`, or `LOW`).
- Simulates destination delay from 40 journey, infrastructure, operating, and weather-related features.
- Exposes health, metrics, and train-catalog endpoints for operations and demos.
- Uses bounded provider caching, retries, request IDs, and an in-memory per-client rate limiter.

## Repository layout

```text
backend/
  main.py                    FastAPI app, routes, auth, middleware
  eta.py                     Cascading ETA calculations
  geo.py                     Route, position, and station helpers
  journey_model.py           Journey request validation/typing
  model_serving.py           Live model loading and feature preparation
  providers.py               RailRadar/Open-Meteo clients and caching
  runtime.py                 Cache, rate limiter, and metrics primitives
  simulator.py               Offline timetable, route, and GPS simulator
  stats.py                   Historical segment statistics
  train_journey_model.py     Optional journey-model training script
  model/                     Model binaries and serving metadata
  dataset/                   Training data and data dictionary
frontend/
  index.html                 Dashboard markup
  app.js                     Dashboard behavior and API calls
  styles.css                 Dashboard styles
tests/                       Offline unittest suite
Dockerfile                   Production container
docker-compose.yml           Local container orchestration
Procfile                    Generic PaaS start command
render.yaml                 Render deployment manifest
```

## Requirements

- Python 3.13 (the Dockerfile and Render manifest use 3.13)
- `pip` and `venv`
- Docker Desktop, if using Docker
- Optional: a RailRadar API key for live data
- A browser with network access to the CDN-hosted Tailwind, Leaflet, Lucide, and map tiles used by the frontend

## Run locally

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open:

- Dashboard: <http://127.0.0.1:8000/> or <http://127.0.0.1:8000/app>
- Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>
- OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>

The dashboard starts with train `12919`. Use the quick-select buttons or enter any integer from `1` to `99999`; unknown numbers are handled by the simulator.

## Configuration

Configuration is read from environment variables and from `backend/.env` (or a project-root `.env`). Do not commit either file.

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAILRADAR_API_KEY` | empty | Enables live RailRadar train/route requests. Empty, `test-key`, and values beginning with `your_` use simulation fallback. |
| `REQUIRE_API_KEY` | `false` | Protects `/predict`, `/predict-journey`, and `/metrics` with `X-API-Key`. |
| `APP_API_KEY` | unset | Expected value of `X-API-Key` when `REQUIRE_API_KEY=true`; startup fails if it is missing. |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins, for example `https://dashboard.example.com,http://localhost:3000`. |
| `REQUEST_RATE_LIMIT_PER_MINUTE` | `120` | Maximum requests per client per rolling minute for protected routes. |
| `JOURNEY_MODEL_URL` | unset | HTTPS release-asset URL used to download the optional ignored journey model at startup. |

Example `.env`:

```dotenv
RAILRADAR_API_KEY=replace_with_your_key
REQUIRE_API_KEY=true
APP_API_KEY=replace_with_an_app_key
CORS_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
REQUEST_RATE_LIMIT_PER_MINUTE=120
```

Restart Uvicorn after changing environment values. Check `/health` to see `provider_mode` (`LIVE` or `SIMULATION_FALLBACK`) and whether the journey model is loaded.

To serve the optional journey simulator in a deployment, upload `journey_delay_model.txt` as a private/public release asset or to a model registry and set `JOURNEY_MODEL_URL` to its HTTPS download URL. The app downloads it only when the local artifact is absent; `/health` reports its artifact status. Do not commit the binary or credentials.

The bundled browser frontend does not send an API-key header. Keep `REQUIRE_API_KEY=false` for the dashboard as-is, or update the frontend/reverse proxy to inject `X-API-Key` before enabling authentication.

## API reference

All request and response schemas are also available in Swagger at `/docs`. Unless authentication is enabled, no API key is needed.

### `GET /` and `GET /app`

Serve the frontend when requested as HTML. A non-browser request to `/` returns service metadata.

### `GET /trains`

Returns the built-in popular-train catalog used by the dashboard. The catalog currently includes trains such as `12919`, `12002`, `22436`, `12424`, and `12952`, with names, endpoints, stop counts, and distance.

### `GET /health`

Returns service status, loaded live-model features, journey-model availability, provider mode, cache counters, and the configured rate limit. This endpoint is suitable for a load-balancer health check.

### `GET /metrics`

Returns aggregate request/provider counters and cache statistics. When `REQUIRE_API_KEY=true`, send `X-API-Key`.

```bash
curl http://127.0.0.1:8000/metrics
curl -H 'X-API-Key: replace_with_an_app_key' http://127.0.0.1:8000/metrics
```

### `POST /predict`

Predict the next-station delay and upcoming ETAs.

Request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"train":12919}'
```

When authentication is enabled, add `-H 'X-API-Key: replace_with_an_app_key'`.

The response includes `current_station`, `next_station`, current and predicted delay minutes, segment progress and source, position and source, historical statistics, weather availability, `next_station_eta`, `upcoming_stations`, route GeoJSON, model features, and `status: "success"`. At a terminal station there is no next-station inference; the current delay is returned and `model.prediction_skipped` is `true`.

Possible errors include `400` for invalid/insufficient route data, `401` for a missing or invalid API key, `429` when the rate limit is exceeded, `502` for an external provider failure, and `500` for an unexpected service error.

### `POST /predict-journey`

Runs the separately trained destination-delay model. The `features` object must contain exactly the 40 fields below: no missing or extra fields are accepted.

```json
{
  "features": {
    "train_number": 12919,
    "train_type": "Superfast Express",
    "year": 2024, "month": 8, "day_of_week": 1,
    "departure_hour": 12, "is_weekend": 0, "is_night_departure": 0,
    "is_peak_hour": 1, "is_festival_season": 0, "season": "Monsoon",
    "zone": "Western Railway (WR)", "zone_abbr": "WR",
    "source_station_category": "A1", "destination_station_category": "A1",
    "distance_km": 1619, "num_scheduled_stops": 11,
    "scheduled_travel_hours": 28.25, "track_doubled": 1, "is_hdn_route": 1,
    "traction_type": "Electric (25kV AC)", "is_electrified": 1,
    "psr_count": 0, "is_circular_route": 0, "is_monsoon_season": 1,
    "is_fog_risk": 0, "fog_risk_score": 0, "zone_fog_index": 0.1,
    "zone_congestion_index": 0.7, "season_severity_score": 0.4,
    "loco_age_years": 6, "coach_age_years": 5, "has_lhb_coaches": 1,
    "is_rake_shared": 0, "maintenance_score": 0.85,
    "seat_utilisation_pct": 78, "is_overloaded": 0,
    "late_incoming_rake": 0, "is_special_train": 0,
    "route_historical_ontime_pct": 82
  }
}
```

Categorical values must match `backend/model/journey_delay_model_config.json`. Numeric values must be finite. A successful response returns predicted destination delay, the 15-minute late threshold, a boolean `is_predicted_delayed`, model name, and stored validation metadata. If the journey artifacts are absent, the endpoint returns `503`.

## Models and data

The live next-station model is `backend/model/champion_model.txt`. Its 13 serving features are listed in `backend/model/model_features.json`; category vocabularies are in `station_categories.json`; historical segment aggregates are in `segment_stats.csv`.

The optional journey model is `journey_delay_model.txt`, with schema and category validation in `journey_delay_model_config.json` and evaluation results in `journey_delay_validation.json`. It is trained with a time-based split: journeys from 2018–2023 are used for training and unseen 2024 journeys for validation. Leakage-prone identifiers, departure date, post-journey cause, and target columns are excluded.

The large `backend/dataset/ir_train.csv` and related files are intentionally ignored by Git. To retrain the journey model, place `ir_train.csv` at `backend/dataset/ir_train.csv` and run:

```bash
python backend/train_journey_model.py
```

The script writes the model, config, and validation JSON into `backend/model/`. Retraining is optional for normal application use; any available serving artifacts are loaded automatically at startup.

## Tests

Run the offline unittest suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests cover request validation, endpoint behavior, mocked providers, journey feature preparation, ETA logic, caching, rate limiting, and authentication. Provider calls are mocked, so the suite does not require a RailRadar key.

GitHub Actions runs this suite on Ubuntu, macOS, and Windows. It also verifies that the LightGBM model has LF line endings, preventing Windows checkouts from reintroducing the model-loading issue.

## Docker

Build and start the service:

```bash
docker compose up --build -d
```

Then visit <http://localhost:8000>. View logs with `docker compose logs -f api` and stop it with `docker compose down`. The container runs as a non-root user and exposes a `/health` Docker health check.

To build without Compose:

```bash
docker build -t railpulse-ai .
docker run --rm -p 8000:8000 -e RAILRADAR_API_KEY=your_key railpulse-ai
```

## Cloud deployment

- Render: create a web service from this repository; `render.yaml` supplies the build and start commands. Add `RAILRADAR_API_KEY` as a secret environment variable if live mode is required.
- Railway, Heroku, or another Procfile-compatible host: use `Procfile`, which binds to the platform-provided `PORT` (default `8000`).
- Any Docker host: deploy the included `Dockerfile` and publish container port `8000`.

For production, set a restrictive `CORS_ORIGINS`, enable `REQUIRE_API_KEY`, use a strong secret `APP_API_KEY`, and put the service behind HTTPS and a reverse proxy. The in-memory cache, rate limiter, and metrics are process-local; use a shared gateway or external store if running multiple replicas.

## Troubleshooting

- `journey_model_loaded` is `false` or `/predict-journey` returns `503`: verify both `backend/model/journey_delay_model.txt` and `journey_delay_model_config.json` exist and are readable.
- `/health` reports `SIMULATION_FALLBACK`: `RAILRADAR_API_KEY` is empty, a placeholder, or the live provider failed. This is expected for offline demos.
- Browser page loads without styling or a map: the frontend loads Tailwind, Leaflet, Lucide, fonts, and map tiles from public CDNs, so allow outbound browser access.
- `401`: set the `X-API-Key` header to the value of `APP_API_KEY`, or disable `REQUIRE_API_KEY` for local development.
- `429`: wait for the rolling minute window or increase `REQUEST_RATE_LIMIT_PER_MINUTE`.
- Port already in use: run `uvicorn backend.main:app --reload --port 8001` and open port `8001`.

## License and project status

This repository is an SIH Railway prototype/reference implementation. Add the project’s preferred license and production data-provider terms before public redistribution. RailRadar, Open-Meteo, Esri map tiles, Google Fonts, Tailwind CDN, Leaflet, and Lucide are third-party services/libraries with their own terms and availability limits.
