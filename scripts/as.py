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

import sys
sys.path.append(os.path.abspath(os.path.join('../scripts')))
from data_generator import make_audio_gen
# from train import train
from models import model_2
from char_map import char_map, index_map


TRAIN_CORPUS = "./data/train_corpus.json"
VALID_CORPUS = "./data/train_corpus.json"
# VALID_CORPUS = "data_stbbl/valid_corpus.json"


MFCC_DIM = 13
SPECTOGRAM = False
EPOCHS = 2
MODEL_NAME = "model_2_stbbli11"

################ Reminder MINI_BATCH_SIZE=250 in previous notebooks
MINI_BATCH_SIZE = 7

SORT_BY_DURATION=False
MAX_DURATION = 10.0

audio_gen = make_audio_gen(TRAIN_CORPUS, VALID_CORPUS, spectrogram=False, mfcc_dim=MFCC_DIM,
                           minibatch_size=MINI_BATCH_SIZE, sort_by_duration=SORT_BY_DURATION,
                           max_duration=MAX_DURATION)
# add the training data to the generator
audio_gen.load_train_data()
audio_gen.load_validation_data()

# loaded_model = pickle.load(
#             open("./models/model_2_stbbl.pickle", 'rb'))


# model = model_2(input_dim=13,
#                 filters=200,
#                 kernel_size=11, 
#                 conv_stride=2,
#                 conv_border_mode='valid',
#                 units=250,
#                 activation='relu',
#                 dropout_rate=1,
#                 number_of_layers=5,
#                 output_dim=len(char_map)+1)
# model_name = "model_2_stbbl"
# model.load_weights('models/' + model_name + '.h5')


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
model_name = "model_2_stbbli"
model.load_weights('models/' + MODEL_NAME + '.h5')
predict(audio_gen, 0, 'train', model, False)



print(type(model))

# predict(audio_gen, 0, 'train', model, False)

