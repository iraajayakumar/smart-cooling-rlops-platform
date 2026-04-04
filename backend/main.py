from fastapi import FastAPI

app = FastAPI(title="Smart Cooling Backend Placeholder")

@app.get("/")
def root():
    return {"message": "Backend placeholder is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics_placeholder():
    return {"message": "Metrics endpoint placeholder"}