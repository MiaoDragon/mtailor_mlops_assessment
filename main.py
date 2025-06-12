"""
This provides code for deployment to Cerebrium.
"""

from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager

import onnx
import onnxruntime

from PIL import Image
import numpy as np

from model import OnnxModel
import base64


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.onnx_model = OnnxModel("model.onnx")
    yield
    del app.state.onnx_model

app = FastAPI()

@app.post("/predict")
def predict(request: Request, img: str):
    """
    the image is represented by Base64 data
    """
    onnx_model: OnnxModel = request.app.state.onnx_model
    if not img:
        raise HTTPException(status_code=400, detail="Image data is required")
    if not isinstance(img, str):
        raise HTTPException(status_code=400, detail="Image data must be a Base64 encoded string")
    try:
        image_data = base64.b64decode(img)
    except Exception:
        return {"error": "Invalid Base64 encoding"}
    # check the output of the onnx model
    onnx_output = onnx_model.predict(image_data)  # return a list of probablities for each class
    return {"prediction": onnx_output.tolist()}


@app.get("/health")
def health():
    return "OK"

@app.get("/ready")
def ready():
    return "OK"