import os
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow.keras as keras
import shutil
import time
import pandas as pd

from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify
# from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER']="images"

model = tf.keras.models.load_model("model1_mobilenet.h5", custom_objects={'KerasLayer':hub.KerasLayer})

# Server test functions
@app.route("/")
def hello():
    return 'Hello World'

# Machine Learning Predict and
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        shutil.rmtree('images')
        os.makedirs('images')
        upload_image=request.files['images']
        filepath=os.path.join(app.config['UPLOAD_FOLDER'],upload_image.filename)
        upload_image.save(filepath)

        #path ke gambar+nama filenya
        fname = "images/{}".format(os.listdir('images/')[0])
        #
        df = pd.read_csv("model/label.csv", sep = ";")
        def return_label(array):
            largest = 0
            for x in range(0, len(array)):
                if(array[x] > largest):
                    largest = array[x]
                    y = x
            return y
        # Read the image
        image_size = (244, 244)
        test_image = image.load_img(fname, target_size = image_size)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        label = return_label(result[0])
        if label == 0:
            id = int(str(time.time()).replace('.', '')[3:13])
            label_name = 'Acne and Rosacea'
            recommendation = df.loc[df["Label"] == label_name]["Recommendation"][0]
        elif label == 1:
            id = int(str(time.time()).replace('.', '')[3:13])
            label_name= 'Eczema'
            recommendation = df.loc[df["Label"] == label_name]["Recommendation"][1]
        elif label == 2:
            id = int(str(time.time()).replace('.', '')[3:13])
            label_name= 'Herpes HPV'
            recommendation = df.loc[df["Label"] == label_name]["Recommendation"][2]
        elif label == 3:
            id = int(str(time.time()).replace('.', '')[3:13])
            label_name = 'Psoriasis pictures Lichen Planus'
            recommendation = df.loc[df["Label"] == label_name]["Recommendation"][3]
        elif label == 4:
            id = int(str(time.time()).replace('.', '')[3:13])
            label_name = 'Seborrheic Keratoses'
            recommendation = df.loc[df["Label"] == label_name]["Recommendation"][4]
        os.remove("images/{}".format(upload_image.filename))
        return jsonify(id=id, label=label_name, recommendation=recommendation )
    else:
        return "Internal Server Error, Using Method Get but not run the code"

# Functions for Skinsis
@app.route("/dictionary")
def dictionary():
    f = open('label.json')
    data = json.load(f)
    return data   

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)