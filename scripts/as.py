import pickle
import sys
import os

print(os.getcwd())
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './scripts')))
from prediction import Prediction
from data_generator import make_audio_gen
from predict import predict
from char_map import char_map
from models import model_2

TRAIN_CORPUS = "data/train_corpus.json"
VALID_CORPUS = "data/train_corpus.json"

MFCC_DIM = 13
SPECTOGRAM = False
EPOCHS = 2
MODEL_NAME = "model_2_stbbl"
MINI_BATCH_SIZE = 250

SORT_BY_DURATION=False
MAX_DURATION = 10.0

audio_gen = make_audio_gen(TRAIN_CORPUS, VALID_CORPUS, spectrogram=False, mfcc_dim=MFCC_DIM,
                           minibatch_size=MINI_BATCH_SIZE, sort_by_duration=SORT_BY_DURATION,
                           max_duration=MAX_DURATION)

audio_gen.load_train_data()
audio_gen.load_validation_data()

loaded_model = pickle.load(
            open("./models/model_2_stbbl.pickle", 'rb'))


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
model_name = "model_2_stbbl"
model.load_weights('models/' + model_name + '.h5')


print(type(model))

predict(audio_gen, 0, 'train', model, False)

