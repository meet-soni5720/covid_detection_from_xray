import os
from flask import Flask, redirect, render_template, request, send_from_directory, url_for
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import load_model

import numpy as np
import cv2
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

model = load_model('cv1_220.h5')

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['upload_folder'] = 'uploads'

@app.route('/')
def index():
    return "<h1> Hello world! </h1>"

def allowed_files(filename):
    allowed_extensions = ['jpg', 'jpeg', 'png']
    #abc.jpg --> ['abc', 'jpg']
    ext = filename.split('.')[-1]
    if ext.lower() in allowed_extensions:
        return True
    else:
        return False

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == "":
            return redirect(request.url)
        
        if file:
            if(allowed_files(file.filename)):
                print(os.path.join(app.config['upload_folder'], file.filename))
                file.save(os.path.join(app.config['upload_folder'], file.filename))
            else:
                return redirect(request.url)
            
            image = cv2.imread(os.path.join(app.config['upload_folder'], file.filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (250, 250), interpolation = cv2.INTER_NEAREST)
            image = np.array(image)/255
            x = np.expand_dims(image, axis = 0)

            print(x.shape)
            arr = model.predict(x)[0]
            print(arr)
            y = np.argmax(arr, axis = 0)
            print(y)

            class_name = ['covid19', 'normal', 'pneumonia']
            class_val = class_name[y]
            confidence = arr[y]

            return render_template('predict.html', image_name = file.filename, class_val = class_val, confidence = confidence*100)

    return render_template('predict.html')

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['upload_folder'], filename)

if __name__ == '__main__':
    app.run(debug=True)