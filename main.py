"""
This provides code for deployment to Cerebrium.
"""

from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel

import onnx
import onnxruntime

from PIL import Image
import numpy as np
from io import BytesIO


from model import OnnxModel
import base64


class ImageInput(BaseModel):
    img: str

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     app.state.onnx_model = OnnxModel("model.onnx")
#     yield
#     del app.state.onnx_model

app = FastAPI()

@app.post("/predict")
def predict(request: Request, payload: ImageInput):
    """
    the image is represented by Base64 data
    """
    # onnx_model: OnnxModel = request.app.state.onnx_model
    onnx_model = OnnxModel("model.onnx")
    img = payload.img
    if not img:
        raise HTTPException(status_code=400, detail="Image data is required")
    if not isinstance(img, str):
        raise HTTPException(status_code=400, detail="Image data must be a Base64 encoded string")
    try:
        image_data = base64.b64decode(img)
        image_data = Image.open(BytesIO(image_data))

        # image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)  # decode the image data
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