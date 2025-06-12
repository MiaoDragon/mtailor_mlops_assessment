"""
this calls the model deployed at Cerebrium.
- This should accept the path of the image and return/print the id of the class the image belongs to
- And also accept a flag to run preset custom tests, something like test.py but uses deployed model.
- Add more tests to test the Cerebrium as a platform. Anything to monitor the deployed model.

  App Dashboard: https://dashboard.cerebrium.ai/projects/p-aae8eb0e/apps/p-aae8eb0e-mtailor-ylmiao
│ Endpoints:                                                                                           │
│ POST https://api.cortex.cerebrium.ai/v4/p-aae8eb0e/mtailor-ylmiao/{function_name} 
"""

import sys
import os
import test
import requests


import base64

def encode_image_to_base64(image_path):
    # reference ChatGPT
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_server.py <image_path> | t")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == 't':
        print('************************')
        print('Running preset tests...')
        # run the tests defined in test.py
        test.test_onnx_model()
    else:
        if os.path.exists(arg):
            print('************************')
            print('Running prediction on image:', arg)
            # use POST to call the deployed model

            url = "https://api.cortex.cerebrium.ai/v4/p-aae8eb0e/mtailor-ylmiao/predict"
            headers = {
                "Authorization": "Bearer your-api-key-here",
                "Content-Type": "application/json"
            }

            data = {
                "img": encode_image_to_base64(arg),
                # Add any other inputs your function expects
            }

            response = requests.post(url, json=data, headers=headers)

            if response.ok:
                print("Response:", response.json())
            else:
                print("Error:", response.status_code, response.text)
        else:
            print(f"Error: The file '{arg}' does not exist.")
            sys.exit(1)
