# coding=utf-8
import os
import sys
import numpy as np
import pandas as pd

# Flask utils
from flask import Flask, request, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './scripts')))
from prediction2 import Prediction

# Instantiate prediction class
prediction = Prediction()

# Define a flask app
app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Upload folder for storing files
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    # Main page
    return {
        "status": "sucess",
        "message": "Amharic Speech Recognition API",
    }

@app.route('/predict', methods=['GET', 'POST'])
def handle_upload():
    if request.method == 'POST':
        return prediction.handle_df_upload(request, secure_filename, app)
    elif request.method == 'GET':
        return {"status": "fail", "error": "No Get Route Supported in /predict endpoint"}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33507))
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', debug=True, port=port)
