"""Predict on wav data uploaded from user."""

import sys
import os
import pandas as pd
import tensorflow as tf
import keras
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './scripts')))

class Prediction:

    def __init__(self) -> None:
        """Initilize class."""
        self.MODEL_NAME = "rnn1"
        # An integer scalar Tensor. The window length in samples.
        self.frame_length = 256
        # An integer scalar Tensor. The number of samples to step.
        self.frame_step = 160
        # An integer scalar Tensor. The size of the FFT to apply.
        # If not provided, uses the smallest power of 2 enclosing frame_length.
        self.fft_length = 384

        try:
            pass
        except Exception:
            sys.exit(1)

    def allowed_file(self, filename):
        ALLOWED_EXTENSIONS = {'wav'}
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    
    def encode_single_sample(self, wav_file, label):
        ###########################################
        ##  Process the Audio
        ##########################################
        # 1. Read wav file
        file = tf.io.read_file(wav_file )
        # 2. Decode the wav file
        audio, _ = tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=-1)
        # 3. Change type to float
        audio = tf.cast(audio, tf.float32)
        # 4. Get the spectrogram
        spectrogram = tf.signal.stft(
            audio, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.fft_length
        )
        # 5. We only need the magnitude, which can be derived by applying tf.abs
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        # 6. normalisation
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        ###########################################
        ##  Process the label
        ##########################################
        # 7. Convert label to Lower case
        label = tf.strings.lower(label)
        # 8. Split the label
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        # 9. Map the characters in label to numbers
        label = self.char_to_num(label)
        # 10. Return a dict as our model is expecting two inputs
        return spectrogram, label

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        # Iterate over the results and get back the text
        output_text = []
        for result in results:
            result = tf.strings.reduce_join(self.num_to_char(result)).numpy().decode("utf-8")
            output_text.append(result)
        return output_text


    def inference(self, model, wav_file):
        batch_size = 16
        a = {"key": wav_file, "text": "ያንደኛ ደረጃ ትምህርታቸው ን ጐንደር ተ ም ረዋል"}
        df = pd.DataFrame(a ,index=[0])
        train_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df["key"]), list(df["text"]))
        )
        train_dataset = (
        train_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        pred = model.predict(train_dataset.take(1))
        pred = self.decode_batch_predictions(pred)
        return pred

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
                except Exception as e:
                    print(e)
                    return {"status": "fail", "error": "Failed to make transcribe audio"}
            else:
                return {"status": "fail", "error": "Only wave files are allowed"}
        elif request.method == 'GET':
            return {"status": "fail", "error": "Get request not available yet"}

    def predict_audio(self, full_path):
        model = keras.models.load_model('models/deep-speech-25', compile=False)

        # The set of characters accepted in the transcription.
        characters =[ x for x in """ሀሁሂሄህሆለሉሊላሌልሎሏመሙሚማሜምሞሟረሩሪራሬርሮሯሰሱሲሳሴስሶሷሸሹሺሻሼሽሾሿቀቁቂቃቄቅቆቋበቡቢባቤብቦቧቨቩቪቫቬቭቮቯተቱቲታቴትቶቷቸቹቺቻቼችቾቿኋነኑኒናኔንኖኗኘኙኚኛኜኝኞኟአኡኢኤእኦኧከኩኪካኬክኮኸኹኺኻኼኽኾኰኲኳወዉዊዋዌውዎዘዙዚዛዜዝዞዟዠዡዢዣዤዥዦዧየዩዪያዬይዮደዱዲዳዴድዶዷጀጁጂጃጄጅጆጇገጉጊጋጌግጐጓጔጠጡጢጣጤጥጦጧጨጩጪጫጬጭጮጯጰጱጲጳጴጵጶጷፀፁፂፃፄፅፆፇጸጹጺጻጼጽጾጿፈፉፊፋፌፍፎፏፐፑፒፓፔፕፖ'?! """]
        # characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
        # Mapping characters to integers
        self.char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
        # Mapping integers back to original characters
        self.num_to_char = keras.layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), oov_token="", invert=True
        )

        pred = self.inference(model, full_path)[0]
        
        return pred
        # predictedd = predict(full_path, model, False)
        # predictedd = predictedd.replace("'", '')
        # return predictedd












# if __name__ == "__main__":
#     pred = inference(model, "tr_4_tr01004.wav")
#     print(pred)