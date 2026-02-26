# Hybrid xT Platform (StatsBomb + XGBoost + FastAPI + React)

End-to-end football analytics system that computes action-level expected threat (xT) from a hybrid model:
1. Zone-based Markov chain xT surface (16x12 grid, 192 zones)
2. XGBoost shot-likelihood delta model

Final action xT is:

`0.5 * (zone_xT_end - zone_xT_start) + 0.5 * (shot_prob_end - shot_prob_start)`

## Project Structure

- `ml/`: Data ingestion, feature engineering, model training, hybrid scoring, aggregation
- `backend/`: FastAPI API serving artifacts and player/action endpoints
- `frontend/`: React dashboard with D3 pitch map and validation scatter
- `data/`: Raw and processed parquet outputs
- `artifacts/`: Serialized xT surface + model + metadata
- `docs/`: Methodology and assumptions

## Phase Mapping

- Phase 1: `ml/data_ingestion.py`
- Phase 2: `ml/features.py`
- Phase 3: `ml/markov_xt.py`
- Phase 4: `ml/model.py`
- Phase 5: `ml/hybrid.py`
- Phase 6: `ml/aggregate.py`
- Phase 7: `backend/main.py`
- Phase 8: `frontend/src/*`
- Phase 9: `docs/methodology.md`

## Setup

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run ML pipeline

```bash
python -m ml.pipeline
```

Outputs:
- `data/processed/actions_hybrid_xt.parquet`
- `data/processed/player_stats.parquet`
- `data/processed/team_stats.parquet`
- `artifacts/xt_surface.npy`
- `artifacts/transition_matrix.npy`
- `artifacts/xgboost_shot_model.joblib`
- `artifacts/metadata.json`

### Run FastAPI backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Endpoints:
- `GET /surface`
- `GET /players`
- `GET /player-actions?player_name=...`
- `GET /player-stats?player_name=...`

### Run React frontend

```bash
cd frontend
npm install
npm run dev
```

Optional API override:

```bash
VITE_API_BASE=http://localhost:8000 npm run dev
```

## Validation

Pipeline writes correlation metrics to `artifacts/metadata.json`:
- Pearson/Spearman between player `xt_per_90` and `goals_per_90`
- Pearson/Spearman between player `xt_per_90` and `xg_per_90`

## Notes

- StatsBomb open data contains richer context for selected events; freeze-frame pressure score is computed only when available.
- Game-state reconstruction uses chronological goal accumulation per match.
- Carries are retained only where end location exists.
