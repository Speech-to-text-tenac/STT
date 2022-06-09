import sys
import os
import tensorflow as tf
import mlflow
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../scripts')))
# sys.path.append(os.path.abspath(os.path.join('../scripts')))

print(os.getcwd())

from scripts.data_generator import make_audio_gen
from scripts.train import train
from scripts.models import model_2
from scripts.char_map import char_map, index_map


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=20000)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

if __name__ == '__main__':

    TRAIN_CORPUS = "./notebooks/train3.json"
    VALID_CORPUS = "./notebooks/train3.json"
    # VALID_CORPUS = "data_stbbl/valid_corpus.json"

    MFCC_DIM = 13
    SPECTOGRAM = False
    MODEL_NAME = "model_2_stbbl"

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


    EPOCHS = 5
    MODEL_NAME = "model_2_stbbli"
    params = {'input_dim':13,
                    'filters':200,
                    'kernel_size':11, 
                    'conv_stride':2,
                    'conv_border_mode':'valid',
                    'units':250,
                    'activation':'relu',
                    'dropout_rate':0.4,}
    model = model_2(input_dim=13,
                    filters=200,
                    kernel_size=11, 
                    conv_stride=2,
                    conv_border_mode='valid',
                    units=250,
                    activation='relu',
                    dropout_rate=0.4,
                    number_of_layers=5,
                    output_dim=len(char_map)+1)

    train(audio_gen, input_to_softmax=model, model_name=MODEL_NAME, epochs=EPOCHS, minibatch_size=MINI_BATCH_SIZE)
    # with open("metrics.txt", 'w') as outfile:
    #         outfile.write("model losses: %2.1f%%\n" % hist.losses[EPOCHS-1])
    mlflow.log_param('model_parameters', params)

    mlflow.log_param('input_shape', 13)
    mlflow.log_param('input_rows', 13)
    mlflow.log_param('mini batch size', MINI_BATCH_SIZE)

    mlflow.sklearn.save_model(model, "model")