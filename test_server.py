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
import numpy as np

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
            API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWFhZThlYjBlIiwiaWF0IjoxNzQ5NzYzODc0LCJleHAiOjIwNjUzMzk4NzR9.vqOSacVoIwRD2D8MB4OwJuvAsdk6psbIppV67Gaq0U_lz5KF7Z-vHWNnLu5N1o1O_G_Zcti6WA5B5w_Fnlo70CvLdjzenHfyoV0PWafEz6C-6LadeBvpy-rns54Q6jxP7G0CcwcQxjE_NXvPnYtdYSYRfzgIzmmPrsoC8n1K7Qf88rn43EjlmQ6nAmhK87XQM_pSOsWQUDxOpkRwJ99ma5AGi2WLevKp986-s7WjWgDMoAh6pCQKrj0Bu2ghoj0k2OLsfLXRt_WAF7E7dmQJ2DVE6xmrXRKt2HISSiCFuoLOWa_2o7jaivDmvYBUcEH4kQ_1q-fBhvHq2nIOV8UT4w"
            url = "https://api.cortex.cerebrium.ai/v4/p-aae8eb0e/mtailor-ylmiao/predict"
            headers = {
                "Authorization": "Bearer " + API_KEY,
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
            output = response.json().get('prediction', 'No prediction found')
            print('Prediction:', np.argmax(output))
        else:
            print(f"Error: The file '{arg}' does not exist.")
            sys.exit(1)
