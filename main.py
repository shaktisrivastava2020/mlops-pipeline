
"""FastAPI app entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

import predictor

from router import router

@asynccontextmanager

async def lifespan(app: FastAPI):

    print("Loading production model...")

    try:

        v = predictor.reload()

        print(f"Loaded {v}" if v else "No production model registered yet.")

    except Exception as e:

        print(f"Warning: could not preload model: {e}")

    yield

    print("Shutting down.")

app = FastAPI(

    title="MLOps Pipeline",

    version="0.1.0",

    description="Drift-aware retraining and champion/challenger evaluation for the churn model.",

    lifespan=lifespan,

)

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)

app.include_router(router)

@app.get("/", tags=["system"])

def root():

    return {

        "service": "mlops-pipeline",

        "docs": "/docs",

        "health": "/health",

        "endpoints": [

            "POST /run-pipeline",

            "POST /predict",

            "GET /models",

            "GET /audit/recent",

            "GET /model/info",

        ],

    }

