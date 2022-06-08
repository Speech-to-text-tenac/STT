"""Predict on wav data uploaded from user."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os


class Prediction:

    def __init__(self) -> None:
        """Initilize class."""
        try:
            pass
        except Exception:
            sys.exit(1)

    def allowed_file(self, filename):
        ALLOWED_EXTENSIONS = {'wav'}
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def preprocess(self, path):
        """."""
        try:
            pass
            # return data
        except Exception:
            self.logger.exception(
                'Failed to get Numerical Columns from Dataframe')
            sys.exit(1)

    def handle_df_upload(self, request, secure_filename, app):
        if request.method == 'POST':
            if 'file' not in request.files:
                # flash('No file part')
                return {"status": "fail", "error": "No file part"}
            file = request.files['file']
            if file.filename == '':
                return {"status": "fail", "error": "No file part"}
            if file and self.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                print(filename)
                file_name = 'pred.wav'
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
                full_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], file_name)
                data = self.preprocess(full_path)
                print(data)
                try:
                    results = self.predict(data)
                    dates = data.index.values
                    dates = dates.astype(str).tolist()
                    sales = list(results)
                    data = {
                        "Date": dates,
                        "Sales": sales
                    }
                    return data
                except Exception:
                    return {"status": "fail", "error": "Failed to predict"}
            else:
                return {"status": "fail", "error": "Only wave files are allowed"}
        elif request.method == 'GET':
            return {"status": "fail", "error": "Get request not available yet"}

    def predict(self, df):
        cols = ['StateHoliday', 'Store', 'DayOfWeek', 'Open', 'Promo',
                'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekOfYear']
        loaded_model = pickle.load(
            open("./models/model_2_stbbl.pickle", 'rb'))
        # loaded_model = pickle.load(open("./models/2022-05-26-10-09-06.pkl", 'rb'))
        df = df[cols]
        result = loaded_model.predict(df)
        result = np.exp(result)
        date = df.index.values

        new_df = pd.DataFrame()
        new_df['Date'] = date
        new_df['Predicted Sales'] = result
        print("RESULT:", result)
        # return {"date": date, "sales": result}
        return result
