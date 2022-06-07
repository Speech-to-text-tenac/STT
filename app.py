# coding=utf-8
import sys
import os
import shutil
import glob
import re
import numpy as np
import pandas as pd

# Flask utils
from flask import Flask, request, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Define a flask app
app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


@app.route("/")
def home_view():
    return "<h1>Welcome to STT engine for Amharic language</h1>"

# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return {
#         "status": "sucess",
#         "message": "Hello World"
#     }


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33507))
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', debug=True, port=port)
