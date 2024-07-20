# use fast api taking string as parameter from a get function and converting it to image and then to numpy array and then to a dataframe and then to a prediction and then to a json file and then to a string and then to a response
from flask import Flask, request
from tensorflow.keras.models import load_model
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
import cv2
import pickle
import base64
import os
import json as JSON
from io import BytesIO
from PIL import Image
from typing import List
from pydantic import BaseModel
import tensorflow as tf

app = Flask(__name__)
cors = CORS(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

weights_path = './braintumorfile.h5'
loaded_model = load_model(weights_path)
print("Model loaded successfully.")



def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    image_data = BytesIO(base64.b64decode(encoded_data))
    img = Image.open(image_data)
    return img

@app.route('/',methods=['GET'])
def home():
    return "Hello World"
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
@app.route("/predict", methods=['POST'])
def read_root():
    print("NAAA1")
    data = JSON.loads("""{"image": ["data:image/jpeg;base64,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"]}
    """)
    print("NAAA2 loaded")
    predict_img = []
    for item in data['image']:
        #Decode the base64-encoded image
        image = get_cv2_image_from_base64_string(item)
        image = cv2.resize(image,(150,150))
        predict_img.append(image)

    print("NAAAA3")
    img_array = np.array(predict_img)
    print("NAAAAA4")
    # img_array = img_array.reshape(1,150,150,3)
    res2 = loaded_model.predict(img_array)
    res2 = res2.argmax()
    return {"result":labels[res2]}


if __name__ == '__main__':
    app.run(port=5000)
