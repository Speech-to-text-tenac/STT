"""Load Audio data from a directory."""
import os
import sys

import librosa
import numpy as np
import pandas as pd
from logger import Logger


class LoadData:
    def __init__(self, data_dir, sample_rate=16000, max_duration=10.0,
                 max_samples=None, max_files=None, verbose=True):
        """Load audio data from a directory."""
        try:
            self.logger = Logger("load_data.log").get_app_logger()
            self.logger.info(
                    'Successfully Instantiated LoadData Class Object')
            self.data_dir = data_dir
            self.sample_rate = sample_rate
            self.max_duration = max_duration
            self.max_samples = max_samples
            self.max_files = max_files
            self.verbose = verbose
        except Exception as e:
            self.logger.error(
                    'Failed to Instantiate LoadData Class Object')
            self.logger.error(e)
            sys.exit(1)

    def load_data(self):
        """Load audio data from a directory."""
        try:
            self.logger.info('Loading data from {}'.format(self.data_dir))
            if self.max_samples is not None:
                self.logger.info('Limiting to {} samples'.format(self.max_samples))
            if self.max_files is not None:
                self.logger.info('Limiting to {} files'.format(self.max_files))
            if self.max_duration is not None:
                self.logger.info('Limiting to {} seconds'.format(self.max_duration))
            # Load metadata
            metadata = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))
            # Filter by duration
            if self.max_duration is not None:
                metadata = metadata[metadata.duration <= self.max_duration]
            # Filter by sample count
            if self.max_samples is not None:
                metadata = metadata[metadata.n_samples <= self.max_samples]
            # Filter by file count
            if self.max_files is not None:
                metadata = metadata[metadata.id <= self.max_files]
            # Load audio
            X = []
            for idx, row in metadata.iterrows():
                file_path = os.path.join(self.data_dir, row.fname)
                if self.verbose:
                    self.logger.info('Loading {}'.format(file_path))
                X.append(self.load_audio(file_path))
            return np.array(X)
        except Exception as e:
            self.logger.error('Failed to load data')
            self.logger.error(e)
            sys.exit(1)
