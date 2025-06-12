"""
codebase to test the code/model written. This should test everything one would expect for ML Model deployment.
"""

import torch
from torch import Tensor
from torchvision import transforms
import onnx
import onnxruntime

from PIL import Image
import numpy as np

from pytorch_model import Classifier
from model import OnnxModel

def test_onnx_model():
    """
    compare the deployed prediction with pytorch prediction
    """
    model = Classifier()
    model.load_state_dict(torch.load("pytorch_model_weights.pth"))
    model.eval()

    onnx_model = OnnxModel("model.onnx")
    # check the output of the onnx model
    img = Image.open("./n01667114_mud_turtle.JPEG")
    inp = model.preprocess_numpy(img).unsqueeze(0) 

    onnx_output = onnx_model.predict(img)

    pytorch_output = model.forward(inp).detach().numpy()[0]
    print("ONNX output:", onnx_output.shape)
    print("PyTorch output:", pytorch_output.shape)
    diff = onnx_output - pytorch_output
    print("difference: ", np.max(np.abs(diff)))


if __name__ == "__main__":
    test_onnx_model()