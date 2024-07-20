# use fast api taking string as parameter from a get function and converting it to image and then to numpy array and then to a dataframe and then to a prediction and then to a json file and then to a string and then to a response
from flask import Flask, request
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
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})
app.config['CORS_HEADERS'] = 'Content-Type'

loaded_model = tf.keras.models.load_model("braintumorfile.h5")

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

@app.route('/home',methods=['GET'])
def home():
    return "Hello World"
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
@app.route("/predict", methods=['POST'])
def read_root():
    data = JSON.loads(request.data)
    predict_img = []
    for item in data['image']:
        #Decode the base64-encoded image
        image = get_cv2_image_from_base64_string(item)
        image = cv2.resize(image,(150,150))
        predict_img.append(image)

    img_array = np.array(predict_img)
    img_array = img_array.reshape(1,150,150,3)
    prediction = loaded_model.predict(img_array)
    result = prediction.argmax()
    print(result)
    print(labels[result])

    return {"result":labels[result]}


if __name__ == '__main__':
    app.run(port=5000)
