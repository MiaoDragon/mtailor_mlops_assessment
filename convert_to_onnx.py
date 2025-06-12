"""
codebase to convert the PyTorch Model to the ONNX model
ref: https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
PyTorch version: 2.7.1
"""

import torch
from torch import Tensor
from pytorch_model import Classifier
from torchvision import transforms
import onnx
import onnxruntime

from PIL import Image
import numpy as np
import urllib.request  # for downloading the model


# for adding preprocessing steps to the ONNX model
# class PreprocessClassifier(Classifier):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#     def forward(self, x: Tensor) -> Tensor:
#         x = self.preprocess_numpy(x)
#         return super().forward(x)


def main():
    # download the model from the request url
    urllib.request.urlretrieve("https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth", 'pytorch_model_weights.pth')
    model = Classifier()
    model.load_state_dict(torch.load("pytorch_model_weights.pth"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)  # input shape for the model
    # dummy_input = Image.open("./n01667114_mud_turtle.JPEG")
    # dummy_input = transforms.ToTensor()(dummy_input).unsqueeze(0)  # convert to tensor and add batch dimension

    # * without adding the preprocessing step
    torch.onnx.export(model, dummy_input, "model.onnx", export_params=True,# opset_version=11,
                      dynamo=True)
    # verify the onnx model
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)


    # * with adding the preprocessing step
    # NOTE: I found that if we want to use preprocess, we can't use transforms functions as it is not able to track
    # correctly, unless we add a custom function. Due to this complication, I decided not to use preprocessing step
    # in the ONNX model.
    # preprocess_model = PreprocessClassifier()
    # preprocess_model.load_state_dict(torch.load("pytorch_model_weights.pth"))
    # preprocess_model.eval()
    # # ONNX export only takes data input in the format of Tensor, so we need to convert the image to Tensor
    # torch.onnx.export(preprocess_model, dummy_input, "preprocess_model.onnx", export_params=True,# opset_version=11,
    #                   dynamo=True)
    
    # # verify the onnx model
    # onnx_model = onnx.load("preprocess_model.onnx")
    # onnx.checker.check_model(onnx_model)


def test_output():
    model = Classifier()
    model.load_state_dict(torch.load("pytorch_model_weights.pth"))
    model.eval()
    # check the output of the onnx model
    img = Image.open("./n01667114_mud_turtle.JPEG")
    inp = model.preprocess_numpy(img).unsqueeze(0) 

    # * without adding the preprocessing step
    onnx_inputs = [inp.numpy(force=True)]
    ort_session = onnxruntime.InferenceSession("model.onnx")
    input_name = ort_session.get_inputs()[0].name
    onnx_output = ort_session.run(None, {input_name: onnx_inputs[0]})[0][0]

    # # * with adding the preprocessing step
    # onnx_inputs = [img]
    # ort_session = onnxruntime.InferenceSession("preprocess_model.onnx")
    # input_name = ort_session.get_inputs()[0].name
    # onnx_output = ort_session.run(None, {input_name: onnx_inputs[0]})[0][0]


    pytorch_output = model.forward(inp).detach().numpy()[0]
    print("ONNX output:", onnx_output.shape)
    print("PyTorch output:", pytorch_output.shape)
    diff = onnx_output - pytorch_output
    print("difference: ", np.max(np.abs(diff)))


if __name__ == "__main__":
    main()
    test_output()