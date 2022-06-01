"""Load Audio data from a directory."""

import os
import sys
import librosa
import json
import pandas as pd
import glob
from logger import Logger


class DataLoader:
    def __init__(self, data_dir, sample_rate=16000, max_duration=10.0,
                 max_samples=None, max_files=None, verbose=True):
        """Load audio data from a directory."""
        try:
            self.logger = Logger("load_data.log").get_app_logger()
            self.logger.info(
                'Successfully Instantiated DataLoader Class Object')
            self.data_dir = data_dir
            self.sample_rate = sample_rate
            self.max_duration = max_duration
        except Exception as e:
            self.logger.error(
                'Failed to Instantiate LoadData Class Object')
            self.logger.error(e)
            sys.exit(1)

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
            for i, file in enumerate(os.listdir(self.data_dir)):
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

    def get_wav_files(self, path: str) -> list:
        """Get the wav files from a folder.

        Args:
            path (str): Path to the folder

        Returns:
            list: List of wav files
        """
        try:
            self.logger.info('Getting wav files')
            wav_files = glob(path)
            self.logger.info('Successfully got wav files')
            return wav_files
        except Exception as e:
            self.logger.error('Failed to get wav files')
            self.logger.error(e)
            sys.exit(1)

    def load_transcription(self, path: str, dest_path: str) -> dict:
        """Load transcription data"""

        audio_path = []
        text = []
        duration = []
        try:
            with open(path) as fp:
                Lines = fp.readlines()
                for line in Lines:
                    valid_json = {}
                    val = line.split(' ')[1:]
                    val = ' '.join(val)
                    # Remove any new line character
                    val = val.replace("\n", "").strip()
                    path = line.split(' ')[0]

                    path = '../data/AMHARIC/data/train/wav/' + path + '.wav'
                    audios = self.get_wav_files(
                        '../data/AMHARIC/data/train/wav/*.wav')
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
                    with open(dest_path, 'a', encoding='utf-8') as fp:
                        fp.write(json.dumps(valid_json, ensure_ascii=False))
                        fp.write("\n")
            self.logger.info('Successfully loaded transcription data')
            self.logger.info(
                'Total number of files: {}'.format(len(audio_path)))
            return audio_path, text, duration
        except Exception as e:
            self.logger.error('Failed to load transcription data')
            self.logger.error(e)
            sys.exit(1)

    def generate_meta_data(self, path, dest_path):
        """Generate meta data csv"""

        try:
            self.logger.info('Generating meta data csv')
            audio_path, text, duration = self.load_transcription(path)
            data = pd.DataFrame(
                {'key': audio_path, 'text': text, 'duration': duration})
            data.to_csv(dest_path, index=False)
            self.logger.info('Successfully generated meta data csv')
        except Exception as e:
            self.logger.error('Failed to generate meta data csv')
            self.logger.error(e)
            sys.exit(1)

    def read_csv(self, csv_file) -> pd.DataFrame:
        """Csv file reader to open and read csv files into a panda dataframe.
        Args:
        -----
        csv_file: str - path of a json file

        Returns
        -------
        dataframe containing data extracted from the csv file"""
        return pd.read_csv(csv_file)
