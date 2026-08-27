"""
================================================================================
  FASTAPI PRODUCTION BACKEND FOR BLAST VIBRATION PREDICTION & EWMA RETRAINING
================================================================================
  Project   : Autonomous ML Framework for Blast-Induced Ground Vibration
  Institute : IIT (BHU) Varanasi | Department of Mining Engineering
  Run       : uvicorn app_fastapi:app --reload --port 8000
  Docs      : http://127.0.0.1:8000/docs  (Swagger Interactive UI)
================================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import local HybridPPVModel and EWMA utilities
from model_utils import HybridPPVModel
from ewma import EWMAHybridModel, load_real_data, engineer_features

# Initialize FastAPI App
app = FastAPI(
    title="Mining AI: Autonomous Blast Vibration Prediction API",
    description=(
        "Production-grade REST API backend for dynamic Peak Particle Velocity (PPV) "
        "prediction and autonomous EWMA model retraining. Developed at IIT (BHU) Varanasi."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration (React, Vue, Web/Mobile Apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global EWMA Model Instance
MODEL_FILE = 'models/ewma_hybrid_model.pkl'
DATA_FILE = 'results/real_field_data.csv'
model_instance: Optional[EWMAHybridModel] = None


# ══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC SCHEMAS (DATA VALIDATION)
# ══════════════════════════════════════════════════════════════════════════════

class BlastInput(BaseModel):
    Distance: float = Field(..., gt=0, description="Distance from blast to monitoring station (meters)", example=400.0)
    Q: float = Field(..., gt=0, description="Maximum charge per delay (kg)", example=35.0)
    No_of_Holes: float = Field(..., gt=0, description="Number of blast holes", example=100.0)
    Depth: float = Field(..., gt=0, description="Hole depth (meters)", example=6.0)
    Spacing: float = Field(..., gt=0, description="Spacing between holes (meters)", example=5.0)
    No_of_Rows: Optional[float] = Field(default=4.0, description="Number of rows of blast holes", example=4.0)
    Seam_location: Optional[str] = Field(default="Seam Top-5", description="Geological seam location", example="Seam Top-5")

class BlastUpdate(BlastInput):
    PPV_Actual: float = Field(..., gt=0, description="Measured Peak Particle Velocity from geophone (mm/s)", example=4.826)


# ══════════════════════════════════════════════════════════════════════════════
#  LIFESPAN / INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def load_or_train_model():
    global model_instance
    try:
        if os.path.exists(MODEL_FILE):
            model_instance = EWMAHybridModel.load(MODEL_FILE)
            print(f"[FASTAPI] Loaded EWMA model from {MODEL_FILE} ({len(model_instance.data)} rows)")
        else:
            print("[FASTAPI] Model file not found. Initializing model from field data...")
            df = load_real_data(DATA_FILE)
            model_instance = EWMAHybridModel(lam=0.20)
            model_instance.fit(df)
            model_instance.save(MODEL_FILE)
    except Exception as e:
        print(f"[FASTAPI WARNING] Failed to load pickled model ({e}). Initializing fresh model...")
        df = load_real_data(DATA_FILE)
        model_instance = EWMAHybridModel(lam=0.20)
        model_instance.fit(df)


# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/health", summary="Health Check")
def health_check():
    return {
        "status": "online",
        "service": "Autonomous Blast Vibration Prediction API",
        "institution": "IIT (BHU) Varanasi",
        "dataset_rows": len(model_instance.data) if model_instance else 0
    }


@app.post("/api/v1/predict", summary="Predict PPV for Blast Parameters")
def predict_ppv(blast: BlastInput):
    """
    Predict Peak Particle Velocity (PPV) for given blast design parameters.
    Returns USBM physics prediction, Hybrid GBR residual prediction, and DGMS safety status.
    """
    if not model_instance or not model_instance.is_fitted:
        raise HTTPException(status_code=500, detail="Model engine not initialized.")

    row_dict = blast.dict()
    res = model_instance.predict_custom(row_dict)

    # DGMS Compliance Check (Threshold = 10.0 mm/s for structures)
    ppv_pred = res['ppv_predicted']
    dgms_limit = 10.0
    is_safe = ppv_pred <= dgms_limit

    # Calculate max allowable charge Q_max for safe PPV
    q_max_safe = ((blast.Distance / (dgms_limit / model_instance.k)**(1.0 / model_instance.n_phys)))**2

    return {
        "input_parameters": row_dict,
        "predictions": {
            "ppv_hybrid_predicted_mm_s": round(ppv_pred, 4),
            "ppv_physics_baseline_mm_s": round(res['physics_ppv'], 4),
            "ml_residual_correction_mm_s": round(res['residual_correction'], 4),
            "scaled_distance_sd": round(res['scaled_distance'], 4),
        },
        "safety_assessment": {
            "dgms_threshold_mm_s": dgms_limit,
            "status": "SAFE" if is_safe else "EXCEEDS DGMS THRESHOLD",
            "max_allowable_charge_q_max_kg": round(q_max_safe, 2)
        }
    }


@app.post("/api/v1/update_blast", summary="Ingest New Blast Data & Trigger Autonomous EWMA Retraining")
def update_blast(blast: BlastUpdate):
    """
    Ingests a newly recorded blast event, appends it to the dataset,
    recalculates EWMA sample weights, and retrains the Hybrid GBR model autonomously.
    """
    if not model_instance:
        raise HTTPException(status_code=500, detail="Model engine not initialized.")

    row_dict = blast.dict()

    # Predict before update (true out-of-sample prediction)
    res = model_instance.predict_row(row_dict)
    model_instance.save(MODEL_FILE)

    # Trigger DVC pipeline autonomous tracking & retraining
    try:
        import subprocess
        subprocess.Popen(["dvc", "repro", "ewma_autonomous_retraining"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dvc_status = "DVC pipeline tracked dataset update and triggered autonomous stage repro."
    except Exception as e:
        dvc_status = f"Local EWMA retrained (DVC background status: {e})"

    h = model_instance.history[-1]

    return {
        "message": "New blast recorded and model autonomously retrained with updated EWMA weights.",
        "dvc_mlops_status": dvc_status,
        "prediction_before_update": {
            "ppv_actual_mm_s": blast.PPV_Actual,
            "ppv_predicted_mm_s": round(res['ppv_predicted'], 4),
            "error_absolute_mm_s": round(res['error_mm'], 4),
            "error_percentage_pct": round(res['error_pct'], 2),
        },
        "retraining_summary": {
            "total_records_in_db": h['n_rows'],
            "weighted_r2": round(h['R2_weighted'], 4),
            "unweighted_r2": round(h['R2_unweighted'], 4),
            "unweighted_mae_mm_s": round(h['MAE_unweighted'], 4),
            "effective_sample_size_n_eff": round(h['effective_n'], 1),
            "last_row_weight": round(model_instance.weights[-1], 6)
        }
    }


@app.get("/api/v1/history", summary="Fetch Recorded Blast History & Predictions")
def get_history(limit: int = Query(default=50, ge=1, le=500)):
    """
    Returns the latest recorded blast parameters and model evaluations.
    """
    if not model_instance:
        raise HTTPException(status_code=500, detail="Model engine not initialized.")

    df = model_instance.data.tail(limit).copy()
    df = df.loc[:, ~df.columns.duplicated()]

    records = df.to_dict(orient="records")
    return {
        "count": len(records),
        "data": records
    }


@app.get("/api/v1/model_status", summary="Get Current Model Status & Memory Horizon")
def get_model_status():
    """
    Returns the current EWMA model status, memory half-life, and decay weights.
    """
    if not model_instance:
        raise HTTPException(status_code=500, detail="Model engine not initialized.")

    h = model_instance.history[-1] if model_instance.history else {}
    return {
        "decay_factor_lambda": model_instance.lam,
        "total_records": len(model_instance.data),
        "effective_sample_size_n_eff": round(h.get('effective_n', 9.0), 2),
        "half_life_blasts": 3.1,
        "horizon_90pct_weight_blasts": 11,
        "usbm_constants": {
            "k": model_instance.k,
            "n": model_instance.n_phys
        },
        "current_metrics": {
            "weighted_r2": h.get('R2_weighted'),
            "unweighted_mae_mm_s": h.get('MAE_unweighted')
        }
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_dashboard():
    """Serves an embedded web dashboard for field engineers."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Blast Vibration Prediction API | IIT (BHU)</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f6f9; margin: 0; padding: 20px; color: #333; }
            .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
            h1 { color: #1B4F72; border-bottom: 3px solid #27AE60; padding-bottom: 10px; margin-top: 0; }
            .badge { background: #27AE60; color: white; padding: 4px 10px; border-radius: 12px; font-size: 14px; font-weight: bold; }
            .btn { display: inline-block; background: #1B4F72; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold; margin-top: 15px; }
            .btn:hover { background: #154360; }
            .card { background: #f8f9fa; border-left: 4px solid #1B4F72; padding: 15px; margin: 20px 0; border-radius: 4px; }
            code { background: #eef2f5; padding: 2px 6px; border-radius: 4px; color: #c7254e; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Autonomous Mining AI Backend <span class="badge">v2.0 FastAPI</span></h1>
            <p><strong>Department of Mining Engineering, IIT (BHU) Varanasi</strong></p>
            <p>Case Study: <em>Bhanegaon Opencast Mine (Wardha Valley Coalfield)</em></p>
            
            <div class="card">
                <h3>⚡ Active REST API Microservice</h3>
                <p>FastAPI production backend providing real-time Peak Particle Velocity (PPV) prediction and autonomous EWMA dynamic model retraining.</p>
                <a href="/docs" class="btn">🚀 Open Interactive Swagger API Docs (/docs)</a>
            </div>

            <h3>Available Endpoints</h3>
            <ul>
                <li><code>POST /api/v1/predict</code> — Predict PPV and check DGMS compliance</li>
                <li><code>POST /api/v1/update_blast</code> — Ingest new blast row and trigger EWMA refitting</li>
                <li><code>GET /api/v1/model_status</code> — Fetch effective sample size ($N_{\text{eff}}$) and decay weights</li>
                <li><code>GET /api/v1/history</code> — Retrieve historical blast logs and predictions</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
