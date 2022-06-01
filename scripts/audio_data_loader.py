import sys
import os
import librosa
from librosa.core import audio
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
import librosa  # for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile  # for audio processing
import warnings

from logger import Logger
sys.path.insert(0, '../logger/')
sys.path.insert(0, '../scripts/')
sys.path.append(os.path.abspath(os.path.join('..')))

app_logger = Logger("../logs/audio_loader.log").get_app_logger()


class AudioLoad:

    def __init__(self) -> None:
        self.logger = Logger("../logs/audio_loader.log").get_app_logger()

    def loaderTrans(self, filename: str):
        '''
                # Loads the audio file and returns the audio data and sample rate
                # @param filename: The path to the audio file
                # @return: The audio data and sample rate
                #
                '''
        name_to_text = {}
        with open(filename, encoding="utf-8") as f:
            f.readline()
            for line in f:
                name = line.split("</s>")[1]
                name = name.replace('(', '')
                name = name.replace(')', '')
                name = name.replace('\n', '')
                name = name.replace(' ', '')
                text = line.split("</s>")[0]
                text = text.replace("<s>", "")
                name_to_text[name] = text
                self.logger.info(f"Training data loaded: {name}")
        return
