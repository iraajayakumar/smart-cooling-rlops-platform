# RL Service Runtime Contract

## Current DevOps assumptions
- Runtime: Python 3.11
- Serving mode: REST API service
- Framework: FastAPI
- ASGI server: Uvicorn
- Container port: 8001
- Entrypoint: `main:app`
- Start command:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port 8001
  ```

## Expected environment variables
- `MODEL_PATH=/app/models/model.pkl`
- `INFERENCE_MODE=api`
- `LOG_LEVEL=INFO`

## Expected endpoints
- `GET /`
- `GET /health`
- `GET /metrics` (placeholder)
- `POST /predict`

## Expected request contract
```json
{
  "temperature": 29.5,
  "workload": 0.82
}
```

## Expected response contract
```json
{
  "temperature": 29.5,
  "workload": 0.82,
  "cooling": 1
}
```

## Placeholder status
If the final RL inference module is not yet integrated, a temporary placeholder API may be used only for Docker, Compose, monitoring, and interface validation.