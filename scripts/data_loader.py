"""Load Audio data from a directory."""

import json
import os
import sys
from isort import file

import librosa
import pandas as pd
import random
import numpy as np
from glob import glob
import tensorflow
import tensorflow as tf
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from tensorflow.keras import backend as K
from logger import Logger
from model import STTModel


class DataLoader:
    def __init__(self, data_dir, sample_rate=16000, max_duration=10.0, minibatch_size=250,
                 max_samples=None, max_files=None, verbose=True, window=20, max_freq=8000):
        """Load audio data from a directory."""
        try:
            self.logger = Logger("load_data.log").get_app_logger()
            self.logger.info(
                'Successfully Instantiated DataLoader Class Object')
            self.data_dir = data_dir
            self.sample_rate = sample_rate
            self.max_duration = max_duration
            self.minibatch_size = minibatch_size
            self.feat_dim = int(0.001 * window * max_freq) + 1
            self.window = window
            self.max_freq = max_freq
            self.feats_mean = np.zeros((self.feat_dim,))
            self.feats_std = np.ones((self.feat_dim,))
            self.cur_train_index = 0
            self.cur_valid_index = 0
            self.spectrogram = False
            self.rng = random.Random(123)
            self.mfcc_dim = 13,
            # self.feat_dim = calc_feat_dim(window, max_freq)
        except Exception as e:
            self.logger.error(
                'Failed to Instantiate LoadData Class Object')
            self.logger.error(e)
            sys.exit(1)

    def get_file_path(self):
        """Return the file path"""
        return self.data_dir

    def load_audios(self, mono: bool, no_of_audios: int = 100) -> tuple:
        """Load the audio files from a folder.

        Args:
            mono (bool): whether to load the audio as mono or not
            no_of_audios (int): Number of files to load

        Returns:
            tuple: Dictionary of sampled audios and Maximum duration of audios
        """
        try:
            self.logger.info(
                'Loading audio files')
            audio_data = {}
            max_duration = 0
            for i, file in enumerate(os.listdir(self.data_dir + 'train/wav/')):
                if i > no_of_audios:
                    break
                sampled_audio, sample_rate = librosa.load(
                    self.data_dir+file, sr=self.sample_rate, mono=mono)
                max_duration = max(len(sampled_audio) /
                                   sample_rate, max_duration)
                audio_data[file.split('.')[0]] = sampled_audio

            return audio_data, max_duration
        except Exception as e:
            self.logger.error('Failed to load data')
            self.logger.error(e)
            sys.exit(1)

    def get_wav_files(self) -> list:
        """Get the wav files from a folder.

        Args:
            path (str): Path to the folder

        Returns:
            list: List of wav files
        """
        try:
            self.logger.info('Getting wav files')
            path = self.data_dir + 'train/wav/'
            path = path + '*.wav'
            wav_files = glob(path)
            self.logger.info('Successfully got wav files')
            return wav_files
        except Exception as e:
            self.logger.error('Failed to get wav files')
            self.logger.error(e)
            sys.exit(1)

    def load_transcription(self, file_path: str, type, dest_path: str = '', save=False) -> dict:
        """Load transcription data"""

        audio_path = []
        text = []
        duration = []
        try:
            with open(file_path + 'text') as fp:
                Lines = fp.readlines()
                for line in Lines:
                    valid_json = {}
                    val = line.split(' ')[1:]
                    val = ' '.join(val)
                    # Remove any new line character
                    val = val.replace("\n", "").strip()
                    path = line.split(' ')[0]

                    # path = '../data/AMHARIC/data/train/wav/' + path + '.wav'
                    path = file_path + 'wav/' + path + '.wav'
                    audios = self.get_wav_files()
                    if path not in audios:
                        continue

                    audio_path.append(path)
                    text.append(val)
                    duration.append(librosa.get_duration(filename=path))
                    valid_json['text'] = val
                    valid_json['key'] = path
                    # GEt the duration of the audio file
                    valid_json['duration'] = librosa.get_duration(
                        filename=path)
                    if save:
                        with open(dest_path, 'a', encoding='utf-8') as fp:
                            fp.write(json.dumps(
                                valid_json, ensure_ascii=False))
                            fp.write("\n")
            if type == 'train':
                self.train_audio_paths = audio_path
                self.train_durations = duration
                self.train_texts = text
                # self.fit_train()
            elif type == 'validation':
                self.valid_audio_paths = audio_path
                self.valid_durations = duration
                self.valid_texts = text
            else:
                raise Exception(
                    "Invalid type to load metadata. Must be train/validation/test")

            # self.train_texts = text
            # self.train_audio_paths = audio_path
            # self.train_durations = duration
            self.logger.info('Successfully loaded transcription data')
            self.logger.info(
                'Total number of files: {}'.format(len(audio_path)))
            return audio_path, text, duration
        except Exception as e:
            self.logger.error('Failed to load transcription data')
            self.logger.error(e)
            sys.exit(1)

    def fit_train(self, k_samples=100):
        """Estimate the mean and std of the features from the training set.
        Params:
            k_samples (int): Use this number of samples for estimation
        """
        k_samples = min(k_samples, len(self.train_audio_paths))
        samples = self.rng.sample(self.train_audio_paths, k_samples)
        feats = [self.featurize(s) for s in samples]
        print("the shape of the features is: ", len(feats))
        print("First one sample is: ", feats[0])
        feats = np.vstack(feats)
        self.feats_mean = np.mean(feats, axis=0)
        self.feats_std = np.std(feats, axis=0)

    def generate_meta_data(self, path, dest_path):
        """Generate meta data csv"""

        try:
            self.logger.info('Generating meta data csv')
            audio_path, text, duration = self.load_transcription(
                path, dest_path, type='train')
            data = pd.DataFrame(
                {'key': audio_path, 'text': text, 'duration': duration})
            data.to_csv(dest_path, index=False)
            print("Meta data creatwd successfully")
            self.logger.info('Successfully generated meta data csv')
        except Exception as e:
            self.logger.error('Failed to generate meta data csv')
            self.logger.error(e)
            sys.exit(1)

    def map_index(self):
        """Map the index of the labels to the correct one."""
        supported = """
            ሀ ሁ ሂ ሄ ህ ሆ
            ለ ሉ ሊ ላ ሌ ል ሎ ሏ
            መ ሙ ሚ ማ ሜ ም ሞ ሟ
            ረ ሩ ሪ ራ ሬ ር ሮ ሯ
            ሰ ሱ ሲ ሳ ሴ ስ ሶ ሷ
            ሸ ሹ ሺ ሻ ሼ ሽ ሾ ሿ
            ቀ ቁ ቂ ቃ ቄ ቅ ቆ ቋ
            በ ቡ ቢ ባ ቤ ብ ቦ ቧ
            ቨ ቩ ቪ ቫ ቬ ቭ ቮ ቯ
            ተ ቱ ቲ ታ ቴ ት ቶ ቷ
            ቸ ቹ ቺ ቻ ቼ ች ቾ ቿ
            ኋ
            ነ ኑ ኒ ና ኔ ን ኖ ኗ
            ኘ ኙ ኚ ኛ ኜ ኝ ኞ ኟ
            አ ኡ ኢ ኤ እ ኦ
            ኧ
            ከ ኩ ኪ ካ ኬ ክ ኮ
            ኳ
            ወ ዉ ዊ ዋ ዌ ው ዎ
            ዘ ዙ ዚ ዛ ዜ ዝ ዞ ዟ
            ዠ ዡ ዢ ዣ ዤ ዥ ዦ ዧ
            የ ዩ ዪ ያ ዬ ይ ዮ
            ደ ዱ ዲ ዳ ዴ ድ ዶ ዷ
            ጀ ጁ ጂ ጃ ጄ ጅ ጆ ጇ
            ገ ጉ ጊ ጋ ጌ ግ ጐ ጓ ጔ
            ጠ ጡ ጢ ጣ ጤ ጥ ጦ ጧ
            ጨ ጩ ጪ ጫ ጬ ጭ ጮ ጯ
            ጰ ጱ ጲ ጳ ጴ ጵ ጶ ጷ
            ፀ ፁ ፂ ፃ ፄ ፅ ፆ ፇ
            ፈ ፉ ፊ ፋ ፌ ፍ ፎ ፏ
            ፐ ፑ ፒ ፓ ፔ ፕ ፖ
            """.split()

        char_map = {}
        char_map["'"] = 0
        char_map["<UNK>"] = 1
        char_map["<SPACE>"] = 2
        index = 3
        for c in supported:
            char_map[c] = index
            index += 1
        index_map = {v+1: k for k, v in char_map.items()}
        return char_map, index_map

    # def ctc_lambda_func(self, args):
    #     """CTC lambda function."""
    #     y_pred, labels, input_length, label_length = args
    #     return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    # def add_ctc_loss(self, input_to_softmax):
    #     """Adds a CTC loss to the model."""
    #     the_labels = tf.keras.Input(
    #         name='the_labels', shape=(None,), dtype='float32')
    #     input_lengths = tf.keras.Input(
    #         name='input_length', shape=(1,), dtype='int64')
    #     label_lengths = tf.keras.Input(
    #         name='label_length', shape=(1,), dtype='int64')
    #     output_lengths = tf.keras.layers.Lambda(
    #         input_to_softmax.output_length)(input_lengths)
    #     # CTC loss is implemented in a lambda layer
    #     loss_out = tf.keras.layers.Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
    #         [input_to_softmax.output, the_labels, output_lengths, label_lengths])
    #     model = Model(
    #         inputs=[input_to_softmax.input, the_labels,
    #                 input_lengths, label_lengths],
    #         outputs=loss_out)
    #     return model

    def text_to_int_sequence(self, text):
        """ Convert text to an integer sequence """
        char_map, _ = self.map_index()
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = char_map['<SPACE>']
            else:
                # print("checking character " + c + " in map:")
                # print(char_map)
                ch = char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def normalize(self, feature, eps=1e-14):
        """ Center a feature using the mean and std
        Params:
            feature (numpy.ndarray): Feature to normalize
        """
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def featurize(self, audio_clip):
        """ For a given audio clip, calculate the corresponding feature
        Params:
            audio_clip (str): Path to the audio clip
        """
        (rate, sig) = wav.read(audio_clip)
        return mfcc(sig, rate, numcep=13)

    def get_batch(self, partition):
        """ Obtain a batch of train, validation, or test data
        """
        if partition == 'train':
            audio_paths, text, _ = self.load_transcription(
                self.data_dir + '/train/', type='train')
            # audio_paths = self.train_audio_paths
            cur_index = self.cur_train_index
            texts = text
        elif partition == 'valid':
            audio_paths, text, _ = self.load_transcription(
                self.data_dir + '/test/', type='validation')
            # audio_paths = self.valid_audio_paths
            cur_index = self.cur_valid_index
            texts = self.valid_texts
        elif partition == 'test':
            audio_paths, text, _ = self.load_transcription(
                self.data_dir + '/test/', type='test')
            # audio_paths = self.valid_audio_paths
            cur_index = self.cur_valid_index
            texts = self.valid_texts
        else:
            raise Exception("Invalid partition. "
                            "Must be train/validation")

        features = [self.normalize(self.featurize(a)) for a in
                    audio_paths[cur_index:cur_index+self.minibatch_size]]

        # calculate necessary sizes
        max_length = max([features[i].shape[0]
                         for i in range(0, self.minibatch_size)])
        max_string_length = max([len(texts[cur_index+i])
                                for i in range(0, self.minibatch_size)])

        # initialize the arrays
        X_data = np.zeros([self.minibatch_size, max_length,
                           self.feat_dim*self.spectrogram + self.mfcc_dim*(not self.spectrogram)])
        labels = np.ones([self.minibatch_size, max_string_length]) * 28
        input_length = np.zeros([self.minibatch_size, 1])
        label_length = np.zeros([self.minibatch_size, 1])

        for i in range(0, self.minibatch_size):
            # calculate X_data & input_length
            feat = features[i]
            input_length[i] = feat.shape[0]
            X_data[i, :feat.shape[0], :] = feat

            # calculate labels & label_length
            label = np.array(self.text_to_int_sequence(texts[cur_index+i]))
            labels[i, :len(label)] = label
            label_length[i] = len(label)

        # return the arrays
        outputs = {'ctc': np.zeros([self.minibatch_size])}
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length
                  }
        return (inputs, outputs)

    def shuffle_data(self, audio_paths, durations, texts):
        """ Shuffle the data (called after making a complete pass through 
            training or validation data during the training process)
        Params:
            audio_paths (list): Paths to audio clips
            durations (list): Durations of utterances for each audio clip
            texts (list): Sentences uttered in each audio clip
        """
        p = np.random.permutation(len(audio_paths))
        audio_paths = [audio_paths[i] for i in p]
        durations = [durations[i] for i in p]
        texts = [texts[i] for i in p]
        return audio_paths, texts, durations

    def shuffle_data_by_partition(self, partition):
        """ Shuffle the training or validation data
        """
        if partition == 'train':
            self.train_audio_paths, self.train_durations, self.train_texts = self.shuffle_data(
                self.train_audio_paths, self.train_durations, self.train_texts)
            self.train_length = len(self.train_texts)
        elif partition == 'valid':
            self.valid_audio_paths, self.valid_durations, self.valid_texts = self.shuffle_data(
                self.valid_audio_paths, self.valid_durations, self.valid_texts)
            self.valid_length = len(self.valid_texts)
        else:
            raise Exception("Invalid partition. "
                            "Must be train/validation")

    # def next_train(self, train_path, train_text, train_duration, mini_batch_size):
    #     """ Obtain a batch of training data
    #     """
    #     self.train_audio_paths = train_path
    #     self.train_texts = train_text
    #     self.train_durations = train_duration
    #     self.minibatch_size = mini_batch_size
    #     while True:
    #         ret = self.get_batch('train')
    #         self.cur_train_index += self.minibatch_size
    #         if self.cur_train_index >= len(self.train_texts) - self.minibatch_size:
    #             self.cur_train_index = 0
    #             self.shuffle_data_by_partition('train')
    #         yield ret

    # def next_valid(self):
    #     """Obtain a batch of validation data."""
    #     while True:
    #         ret = self.get_batch('valid')
    #         self.cur_valid_index += self.minibatch_size
    #         if self.cur_valid_index >= len(self.valid_texts) - self.minibatch_size:
    #             self.cur_valid_index = 0
    #             self.shuffle_data_by_partition('valid')
    #         yield ret

    def read_csv(self, csv_file) -> pd.DataFrame:
        """Csv file reader to open and read csv files into a panda dataframe.
        Args:
        -----
        csv_file: str - path of a json file

        Returns
        -------
        dataframe containing data extracted from the csv file"""
        return pd.read_csv(csv_file)
