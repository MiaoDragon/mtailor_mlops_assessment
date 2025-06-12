"""
model.py with the following classes/functionalities, make their separate classes:
- Onnx Model loading and prediction call
- Pre-processing of the Image [Sample code provided in pytorch_model.py]
"""

import onnxruntime as ort
from PIL import Image
import numpy as np
from pytorch_model import Classifier


class OnnxModel:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.preprocess = ImagePreprocessor()
 
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        input shape: width x height x 3
        """
        inp = self.preprocess.preprocess_numpy(input_data).unsqueeze(0).numpy()
        inputs = {self.input_name: inp}
        outputs = self.session.run([self.output_name], inputs)
        return outputs[0][0]


class ImagePreprocessor(Classifier):
    """
    a direct inheritence of preprocessing functions from Classifier to make sure
    we use the same preprocessing steps as in the PyTorch model.
    """
    pass

if __name__ == "__main__":
    # Example usage
    model_path = "model.onnx"  # Path to your ONNX model
    image_path = "./n01667114_mud_turtle.JPEG"  # Path to the input image

    onnx_model = OnnxModel(model_path)
    image = Image.open(image_path)

    prediction = onnx_model.predict(image)
    print('prediction shape: ', prediction.shape)
    print("Prediction:", prediction)