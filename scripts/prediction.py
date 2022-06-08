"""Predict on wav data uploaded from user."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './scripts')))
from predict import predict
from char_map import char_map
from models import model_2

class Prediction:

    def __init__(self) -> None:
        """Initilize class."""
        self.MODEL_NAME = "model_2_stbbli11"
        try:
            pass
        except Exception:
            sys.exit(1)

    def allowed_file(self, filename):
        ALLOWED_EXTENSIONS = {'wav'}
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
                try:
                    results = self.predict_audio(full_path)
                    return {"status": "success", "message": results}
                except Exception:
                    return {"status": "fail", "error": "Failed to make transcribe audio"}
            else:
                return {"status": "fail", "error": "Only wave files are allowed"}
        elif request.method == 'GET':
            return {"status": "fail", "error": "Get request not available yet"}

    def predict_audio(self, full_path):
        model = model_2(input_dim=13,
                filters=200,
                kernel_size=11, 
                conv_stride=2,
                conv_border_mode='valid',
                units=250,
                activation='relu',
                dropout_rate=1,
                number_of_layers=5,
                output_dim=len(char_map)+1)
        model.load_weights('models/' + self.MODEL_NAME + '.h5')
       
        predictedd = predict(full_path, model, False)
        predictedd = predictedd.replace("'", '')
        return predictedd
