## Deliverables
- convert_to_onnx.py
- model.py
- test.py
- test_server.py

## Steps for running the code
1. download the PyTorch model from the url: https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth
2. in a terminal, run `python.convert_to_onnx.py` to convert the PyTorch to ONNX model. This script also provides testing code for comparing the ONNX prediction with PyTorch prediction.
3. the `model.py` encapsulates the ONNX model and prediction functions. run `python model.py` for testing the code and seeing the ONNX output.
4. the `test.py` provides a testing code for verifying the ONNX model implemented in `model.py` and compares its prediction with the PyTorch model.