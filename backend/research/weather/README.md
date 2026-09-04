# Weather & Environmental Feature Research (V3 Roadmap)

This directory is reserved for offline research, exploratory datasets, and experimental pipelines integrating real-time weather and meteorological conditions into future railway delay models.

> [!NOTE]
> Weather features are strictly isolated from the production backend (`backend/main.py`) and the production V2 model (`champion_model_scheduled_segment_v2.txt`).

## Research Objectives
1. **Open-Meteo Integration**: Dynamic atmospheric data extraction (precipitation, visibility, wind gusts, temperature extremes) along station geographic coordinates.
2. **Monsoon & Fog Regime Analysis**: Evaluating delay propagation spikes under dense fog (North India winter schedules) and heavy rainfall (Konkan / coastal corridors).
3. **Causal Validation**: Ensuring weather feature inclusion does not introduce lookahead leakage or degrade clear-weather prediction accuracy.
