"""
FastAPI server for exposing data analysis and ML predictions.
Run: uvicorn src.api_server:app --reload
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Analyst API",
    description="API for data analysis, processing, and ML predictions",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class HealthResponse(BaseModel):
    status: str
    message: str


class DataStats(BaseModel):
    mean: float
    median: float
    std: float
    min: float
    max: float
    missing: int


class PredictionRequest(BaseModel):
    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    prediction: Any
    confidence: float = None
    timestamp: str


# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "message": "Data Analyst API is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "API is operational"
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and preview CSV file."""
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
            return {
                "filename": file.filename,
                "rows": len(df),
                "columns": df.shape[1],
                "column_names": list(df.columns),
                "preview": df.head(5).to_dict()
            }
        else:
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/stats")
async def get_stats(data: Dict[str, List[float]]):
    """Calculate statistics for numeric data."""
    try:
        stats_dict = {}
        for key, values in data.items():
            arr = np.array(values)
            stats_dict[key] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "missing": int(np.isnan(arr).sum())
            }
        return stats_dict
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using trained model."""
    try:
        # Load model (example - implement your model loading logic)
        # model = pickle.load(open('models/model.pkl', 'rb'))
        # prediction = model.predict([request.features])
        
        return {
            "prediction": "Model prediction",
            "confidence": 0.85,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze")
async def analyze_data(data: Dict[str, List[float]]):
    """Perform basic data analysis."""
    try:
        df = pd.DataFrame(data)
        analysis = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "correlation": df.corr().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        return analysis
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
