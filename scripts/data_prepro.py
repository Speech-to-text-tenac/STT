import math
import random
import sys
import warnings
import pandas as pd
import numpy as np
import librosa  # for audio processing
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display
from pydub import AudioSegment
from scipy.io import wavfile  # for audio processing
# from torchaudio import transforms

warnings.filterwarnings("ignore")


class AudioUtil():
    def __init__(self):
        """Initialize data preprocessing."""
        pass
        # try:
        #     self.logger = Logger("preprocess_data.log").get_app_logger()
        #     self.logger.info(
        #         'Successfully Instantiated DataLoader Class Object')
        # except Exception as e:
        #     self.logger.error(
        #         'Failed to Instantiate LoadData Class Object')
        #     self.logger.error(e)
        #     sys.exit(1)

    def create_meta_data( self, df: pd.DataFrame, column1:str, column2:str):
        df.rename(columns = {0: column1}, inplace = True)
        df[column2] = df[column1].apply(lambda x: x.split("</s>")[1].replace("(", "").replace(")", "").strip())
        df[column1] = df[column1].apply(lambda x: x.split("</s>")[0])
        df[column1] = df[column1].apply(lambda x: x.split("<s>")[1].strip())
        df[column2] = df[column2].apply(lambda x: "data/train/wav/"+x+".wav")
        return df


    def convert_to_stereo(self, audio_path, dest_path, new_channel) -> None:
        """Convert the audio to stereo."""
        try:
            for file in audio_path:
                sound = AudioSegment.from_file(file, format="wav")
                if sound.channels == new_channel:
                    continue
                else:
                    sound = sound.set_channels(new_channel)

                file_name = dest_path + file.split('/')[-1]
                sound.export(file_name, format="wav")
            # self.logger.info("Successfully converted audio to stereo")
        except Exception as e:
            # self.logger.error('Failed to convert to stereo')
            # self.logger.error(e)
            sys.exit(1)

  
    def resample(self, df, column):
        sampled_audio = []
        rates = []
        for i in df[column]:
            audio, rate=librosa.load(i, sr=44100)
            sampled_audio.append(audio)
            rates.append(rate)
        
        return sampled_audio, rates
    
    def resize_audio(self,audios,max_duration):
        resized_audios = {}
        for label in audios:
            resized_audios[label] = librosa.util.fix_length(audios[label],size=int(max_duration*44100))
        return resized_audios

    # def resize_audio(self, audios: dict, max_duration: float) -> dict:
    #     """Extend duration of audio samples to max_duration.

    #     Args:
    #             audios (dict): Dictionary of audio samples.
    #             max_duration (float): The duration to set for the audio samples

    #     Returns:
    #             dict: Dictionary of resized audio samples.
    #     """
    #     try:
    #         # self.logger.info("Resizing audio samples")
    #         resized_audios = {}
    #         for label in audios:
    #             resized_audios[label] = librosa.util.fix_length(
    #                 audios[label], size=int(max_duration*44100))
    #         return resized_audios
    #     except Exception as e:
    #         # self.logger.error('Failed to resize audio')
    #         # self.logger.error(e)
    #         sys.exit(1)
        # ----------------------------
        # Show a widget to play the audio sound
        # ----------------------------

