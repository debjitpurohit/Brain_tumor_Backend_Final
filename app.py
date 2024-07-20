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
    print(request.data)
    data = JSON.loads(request.data)
    print(data)
    predict_img = []
    for item in data['image']:
        #Decode the base64-encoded image
        image = get_cv2_image_from_base64_string(item)
        image = cv2.resize(image,(150,150))
        predict_img.append(image)

    print(predict_img)
    img_array = np.array(predict_img)
    print(img_array)
    # img_array = img_array.reshape(1,150,150,3)
    res2 = loaded_model.predict(img_array)
    # res2 = res2.argmax()
    print("no prblm",res2)
    res2=0
    return {"result":labels[res2]}


if __name__ == '__main__':
    app.run(port=5000)
