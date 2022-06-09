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
import sklearn
from numpy.lib.stride_tricks import as_strided
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


    def augment_audio(self, audios : dict, sample_rate : int) -> dict:
        for name in audios:
            audios[name] = np.roll(audios[name], int(sample_rate/10))
        return audios

    def plot_spec(self, data:np.array,sr:int) -> None:
        '''
        Function for plotting spectrogram along with amplitude wave graph
        '''
        fig, ax = plt.subplots(1,2,figsize=(15,5))
        ax[0].title.set_text(f'Shfiting the wave by Times {sr/10}')
        ax[0].specgram(data,Fs=2)
        ax[1].set_ylabel('Amplitude')
        ax[1].plot(np.linspace(0,1,len(data)), data)

    
    def spectrogram(self, samples, fft_length=256, sample_rate=2, hop_length=128):
        """
        Compute the spectrogram for a real signal.
        The parameters follow the naming convention of
        matplotlib.mlab.specgram
        
        This code was obtained from the notebook provided by @Desmond, one of the tutors at 10Academy batch 5 training

        Args:
            samples (1D array): input audio signal
            fft_length (int): number of elements in fft window
            sample_rate (scalar): sample rate
            hop_length (int): hop length (relative offset between neighboring
                fft windows).

        Returns:
            x (2D array): spectrogram [frequency x time]
            freq (1D array): frequency of each row in x

        Note:
            This is a truncating computation e.g. if fft_length=10,
            hop_length=5 and the signal has 23 elements, then the
            last 3 elements will be truncated.
        """
        assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

        window = np.hanning(fft_length)[:, None]
        window_norm = np.sum(window**2)

        # The scaling below follows the convention of
        # matplotlib.mlab.specgram which is the same as
        # matlabs specgram.
        scale = window_norm * sample_rate

        trunc = (len(samples) - fft_length) % hop_length
        x = samples[:len(samples) - trunc]

        # "stride trick" reshape to include overlap
        nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
        nstrides = (x.strides[0], x.strides[0] * hop_length)
        x = as_strided(x, shape=nshape, strides=nstrides)

        # window stride sanity check
        assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

        # broadcast window, compute fft over columns and square mod
        x = np.fft.rfft(x * window, axis=0)
        x = np.absolute(x)**2

        # scale, 2.0 for everything except dc and fft_length/2
        x[1:-1, :] *= (2.0 / scale)
        x[(0, -1), :] /= scale

        freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

        return x, freqs

    
    

    def mfcc(self, audio, sampling_rate):
        """
        This function computes the Mel frequency cepstral coefficients (MFCCs) of an audio signal

        Precondition:
            librosa is installed in the active environmen

        Args:
            audio: (1D array) The input audio signal
            sampling_rate: (scalar) The sampling rate of the audio signal
        """
        return librosa.feature.mfcc(audio, sr=sampling_rate)


    def plot_mfcc(self, mfccs, sampling_rate):
        """
        This function plots the mfccs of an audio signal

        Precondition:
            librosa is installed in the active environment

        Args:
            mfccs: Mel frequency cepstral coefficient of the audio signal
            sampling_rate: (scalar) The sampling rate of the audio signal
        """
        librosa.display.specshow(mfccs, sr=sampling_rate, x_axis='time')


    def plot_spectrogram(self, vis_spectrogram_feature):
        """
        This function plots a normalized spectogram
        This code was obtained from the notebook provided by @Desmond, one of the tutors at 10Academy batch 5 training


        Args:
        vis_spectogram_feature: 2D array of the spectogram to visualize
        """
            # plot the normalized spectrogram
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(111)
        im = ax.imshow(vis_spectrogram_feature, cmap=plt.cm.jet, aspect='auto')
        plt.title('Spectrogram')
        plt.ylabel('Time')
        plt.xlabel('Frequency')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()


    def spectral_centroids(self, audio, sampling_rate):
        """
        computes the spectral centroid for each frame in an audio signal and the time variable for visualization
        """
        spectral_centroids = librosa.feature.spectral_centroid(audio, sr=sampling_rate)
        frames = range(len(spectral_centroids[0]))
        t = librosa.frames_to_time(frames)
        return spectral_centroids, t

    # Normalising the spectral centroid for visualisation
    def normalize(self, x, axis=0):
        return sklearn.preprocessing.minmax_scale(x, axis=axis)