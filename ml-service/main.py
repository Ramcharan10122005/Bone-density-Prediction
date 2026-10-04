"""FastAPI inference microservice for the two-stage dental implant ML pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pipeline import DEVICE, PipelineModelManager, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml-service")

app = FastAPI(
    title="Dental Implant Segmentation & Bone Quality API",
    description="Two-stage 3D UNet pipeline for Jaw ROI and dental implant segmentation with bone quality classification.",
    version="1.0.0",
)

# CORS middleware allowing all origins for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Initializing models on device: %s...", DEVICE)
    try:
        PipelineModelManager.get_models()
        logger.info("Stage 1 and Stage 2 models loaded successfully into memory.")
    except Exception as e:
        logger.error("Failed to load models during startup: %s", e)


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ml-service",
        "device": str(DEVICE),
        "stage1_loaded": PipelineModelManager._stage1_model is not None,
        "stage2_loaded": PipelineModelManager._stage2_model is not None,
    }


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "Dental Implant Segmentation & Bone Quality API",
        "docs": "/docs",
        "endpoint": "POST /infer",
    }


@app.post("/infer", status_code=status.HTTP_200_OK)
async def infer(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Run two-stage segmentation and bone quality assessment on uploaded CBCT .nrrd file.

    Returns:
    {
      "bone_quality": "D1" | "D2" | "D3" | "D4",
      "implant_volume_voxels": int,
      "roi_mesh": {"vertices": [...], "faces": [...]},
      "implant_mesh": {"vertices": [...], "faces": [...]},
      "mean_intensity": float,
      "crop_bbox": {...},
      "volume_shape": [...]
    }
    """
    filename = file.filename or "upload.nrrd"
    if not (filename.endswith(".nrrd") or filename.endswith(".nhdr")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {filename}. Please upload a .nrrd or .nhdr file.",
        )

    logger.info("Received inference request for file: %s (content_type: %s)", filename, file.content_type)

    temp_file = tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False)
    temp_path = temp_file.name

    try:
        # Stream upload to temp file
        contents = await file.read()
        temp_file.write(contents)
        temp_file.flush()
        temp_file.close()

        logger.info("Saved upload to %s (%d bytes). Running pipeline...", temp_path, len(contents))
        result = run_pipeline(temp_path)
        logger.info(
            "Inference successful: Bone Quality=%s, Implant Volume=%d voxels",
            result["bone_quality"],
            result["implant_volume_voxels"],
        )
        return result

    except Exception as err:
        logger.exception("Inference error while processing %s: %s", filename, err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failed: {err}",
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning("Failed to remove temp file %s: %e", temp_path, e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
