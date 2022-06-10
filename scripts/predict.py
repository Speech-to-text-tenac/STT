
import numpy as np
from tensorflow.keras import backend as K
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './scripts')))
from models import *
from utils import int_sequence_to_text


def featurize(audio_clip):
        """ For a given audio clip, calculate the corresponding feature
        Params:
            audio_clip (str): Path to the audio clip
        """
        (rate, sig) = wav.read(audio_clip)
        return mfcc(sig, rate, numcep=13)

def normalize(feature, eps=1e-14):
    """ Center a feature using the mean and std
        Params:
            feature (numpy.ndarray): Feature to normalize
    """
    feats_mean = np.mean(feature, axis=0)
    feats_std = np.std(feature, axis=0)
    return (feature - feats_mean) / (feats_std + eps)

def predict(audio_path, model, verbose=True):
    """ Print a model's decoded predictions
    Params:
        data_gen: Data to run prediction on
        index (int): Example to visualize
        partition (str): Either 'train' or 'validation'
        model (Model): The acoustic model
    """
    data_point, prediction = predict_raw(audio_path, model)
    output_length = [model.output_length(data_point.shape[0])]
    pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length, greedy=False)[0][0])+1).flatten().tolist()
    predicted = ''.join(int_sequence_to_text(pred_ints)).replace("<SPACE>", " ")
  
    return predicted

def predict_raw(audio_path, model):
    """ Get a model's decoded predictions
    Params:
        data_gen: Data to run prediction on
        index (int): Example to visualize
        partition (str): Either 'train' or 'validation'
        model (Model): The acoustic model
    """

    data_point = normalize(featurize(audio_path))
    
        
    prediction = model.predict(np.expand_dims(data_point, axis=0))
    return (data_point, prediction)
