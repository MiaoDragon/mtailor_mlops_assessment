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
            API_KEY = "eyJraWQiOiJDcEg4RnR0U0xHNWVOQUFBS3pVUWhLVlp4ZUZReWJMaWFkdGlBRjc0WmNrPSIsImFsZyI6IlJTMjU2In0.eyJhdF9oYXNoIjoiTDdRTi1ENUlwX19zOE1fSjJ3bFF5QSIsInN1YiI6IjU2NjA2ZjhiLTY0MDktNGVlYy1iZTVjLWFkN2EwMmQwZThmMCIsImNvZ25pdG86Z3JvdXBzIjpbImV1LXdlc3QtMV9rMm1DWDZYWHJfR29vZ2xlIl0sImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJpc3MiOiJodHRwczpcL1wvY29nbml0by1pZHAuZXUtd2VzdC0xLmFtYXpvbmF3cy5jb21cL2V1LXdlc3QtMV9rMm1DWDZYWHIiLCJwaG9uZV9udW1iZXJfdmVyaWZpZWQiOmZhbHNlLCJjb2duaXRvOnVzZXJuYW1lIjoiR29vZ2xlXzExMTgzMTQzODk5NjIxMTUyMzgwOSIsImdpdmVuX25hbWUiOiJZaW5nbG9uZyIsIm5vbmNlIjoicWN5Q1h0Uks2WXhRdGxGNVVQWUUwUWJQVW9SMnQ2NTBJZV9iZ0xSdGY5LUZyUEF0Rnp1Mmw4U0E3b2xxekhiMWd6R19jOUM2R0o0LWlDMlB3Q0otNVdrZndQaC1EWkR0WmJSNlRiZ1JCTkRHU282QzZnVmFUMGQ1WUtGNzVQMEJ2RjVSc1hsYTZ3RF8ycVBhT1hGWG9kSTNaMjdwR2JxOGl3SDgwejcxQ2c4IiwicGljdHVyZSI6Imh0dHBzOlwvXC9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tXC9hXC9BQ2c4b2NKUlVRNmZzYnQ4MVRGR2VCX1l0UUlveWV3SDdGbEhJSU5QdEpvQTBmTEEtRFVhSEE9czk2LWMiLCJvcmlnaW5fanRpIjoiMWU1NTQ1MTgtZDJkOC00MjA2LThjZGQtZjY4ZjAyMzczYzhjIiwiYXVkIjoiMm9tMHVlbXBsNjl0NGM2ZmM3MHVqc3RzdWsiLCJpZGVudGl0aWVzIjpbeyJ1c2VySWQiOiIxMTE4MzE0Mzg5OTYyMTE1MjM4MDkiLCJwcm92aWRlck5hbWUiOiJHb29nbGUiLCJwcm92aWRlclR5cGUiOiJHb29nbGUiLCJpc3N1ZXIiOm51bGwsInByaW1hcnkiOiJ0cnVlIiwiZGF0ZUNyZWF0ZWQiOiIxNzQ5NzYzODcyNDM2In1dLCJ0b2tlbl91c2UiOiJpZCIsImF1dGhfdGltZSI6MTc0OTc2Mzg3NywiZXhwIjoxNzQ5NzY3NDc3LCJpYXQiOjE3NDk3NjM4NzcsImZhbWlseV9uYW1lIjoiTWlhbyIsImp0aSI6IjEwZDQ3ZDBjLTg4MTItNGM0OC1hOTUxLTQzNmI4MzUyYTk5OSIsImVtYWlsIjoieWluZ2xvbmdtaWFvM0BnbWFpbC5jb20ifQ.urYcmnDX-6DvTsIz3sD8T_yhvfHDnA6nhgP96zwmKNOpJ9Y5e2Ngsyf69RVJvRbHZfBaL8FdmOfZirn0YzvT9kxn8XoR2AWGIRWFzYEWxGsfxiWK6wY3BeA2Wz_TGXMQe0AB5BGXd3wpnv4W-jH_k2WSVtvqJEK-fg7PEo4vMLBbKRImRCJJKRJkVMeeob-X6HSq3jIx6EuieFQtvcvrQB2FStEdn6r6ABusyoYXJh3gUH2jyJ6wGLncjBXMSkx6cZA-7EmFCMjquza6_qDaGV4VcS175gs78TYpG37mTS6GNS-c9k6Bw2l6nsz635L7P233fp3Ed_tXWQ5QlK1rTQ"
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
