# Backend Runtime Contract

## Current DevOps assumptions
- Runtime: Python 3.11
- Framework: FastAPI
- ASGI server: Uvicorn
- Container port: 8000
- Entrypoint: `main:app`
- Start command:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```
- Health endpoint: `/health`
- Metrics endpoint: `/metrics` (placeholder for now)

## Expected files inside `backend/`
- `main.py`
- `requirements.txt`

## Environment variables expected later
- `BACKEND_HOST`
- `BACKEND_PORT`
- `LOG_LEVEL`

## Placeholder status
If the real backend is not integrated yet, a temporary placeholder FastAPI app may exist only to validate Docker build, health checks, and future Compose wiring.