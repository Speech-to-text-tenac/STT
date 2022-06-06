"""Model for STT engine."""
import sys
import os

import tensorflow as tf
import numpy as np
import tensorflow
from logger import Logger
import random
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import pickle

# sys.path.insert(0, os.path.abspath(os.path.join(
#     os.path.dirname(__file__), '../scripts')))

# from data_loader import DataLoader


class STTModel():
    def __init__(self, loader):
        """Initialize STT modeling."""
        try:
            self.logger = Logger("model.log").get_app_logger()
            self.logger.info(
                ' Instantiated STTModel Class Object')
            self.loader = loader
        #    self.loader.load_data()
            file_path = self.loader.get_file_path()
            self.train_audio_paths, self.train_audio_text, self.train_audio_duration = self.loader.load_transcription(
                file_path + 'train/', type='train')
            self.valid_audio_paths, self.valid_audio_text, self.valid_audio_duration = self.loader.load_transcription(
                file_path + 'test/', type='validation')
            # self.fit_train()
            self.cur_train_index = 0
            self.cur_valid_index = 0
            self.window = 20
            self.rng = random.Random(123)
            self.max_freq = 8000
            self.feat_dim = int(0.001 * self.window * self.max_freq) + 1
            self.feats_mean = np.zeros((self.feat_dim,))
            self.feats_std = np.ones((self.feat_dim,))
            self.mfcc_dim = 13
            # self.loader.train_audio_paths = self.train_audio_paths
            # self.loader.train_texts = self.train_audio_text
            # self.loader.train_durations = self.train_audio_duration
            # self.loader.valid_audio_paths = self.valid_audio_paths
            # self.loader.valid_texts = self.valid_audio_text
            # self.loader.valid_durations = self.valid_audio_duration
            # sys.exit(0)
            # Initialize data_loader
            # self.loader = DataLoader(self.data_dir + '/train/wav/')
        except Exception as e:
            self.logger.error(
                'Failed to Instantiate STTModel Class Object')
            self.logger.error(e)
            sys.exit(1)

    def model(self, input_dim, filters, kernel_size, conv_stride,
              conv_border_mode, units, output_dim=29, dropout_rate=0.5, number_of_layers=2,
              cell=tf.keras.layers.GRU, activation='tanh'):
        """Build a deep network for speech."""
        # Main acoustic input
        input_data = tf.keras.Input(name='the_input', shape=(None, input_dim))
        # TODO: Specify the layers in your network
        conv_1d = tf.keras.layers.Conv1D(filters, kernel_size,
                                         strides=conv_stride,
                                         padding=conv_border_mode,
                                         activation='relu',
                                         name='layer_1_conv',
                                         dilation_rate=1)(input_data)
        conv_bn = tf.keras.layers.BatchNormalization(
            name='conv_batch_norm')(conv_1d)

        if number_of_layers == 1:
            layer = cell(units, activation=activation,
                         return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
            layer = tf.keras.layers.BatchNormalization(name='bt_rnn_1')(layer)
        else:
            layer = cell(units, activation=activation,
                         return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
            layer = tf.keras.layers.BatchNormalization(name='bt_rnn_1')(layer)

            for i in range(number_of_layers - 2):
                layer = cell(units, activation=activation,
                             return_sequences=True, implementation=2, name='rnn_{}'.format(i+2), dropout=dropout_rate)(layer)
                layer = tf.keras.layers.BatchNormalization(
                    name='bt_rnn_{}'.format(i+2))(layer)

            layer = cell(units, activation=activation,
                         return_sequences=True, implementation=2, name='final_layer_of_rnn')(layer)
            layer = tf.keras.layers.BatchNormalization(
                name='bt_rnn_final')(layer)

        time_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(output_dim))(layer)
        # TODO: Add softmax activation layer
        y_pred = tf.keras.layers.Activation(
            'softmax', name='softmax')(time_dense)
        # Specify the model
        model = Model(inputs=input_data, outputs=y_pred)
        # TODO: Specify model.output_length
        model.output_length = lambda x: self.cnn_output_length(
            x, kernel_size, conv_border_mode, conv_stride)
        print(model.summary())
        # plot_model(model, to_file='models/model.png', show_shapes=True)
        return model

    # def map_index(self):
    #     """Map the index of the labels to the correct one."""
    #     supported = """
    #         ሀ ሁ ሂ ሄ ህ ሆ
    #         ለ ሉ ሊ ላ ሌ ል ሎ ሏ
    #         መ ሙ ሚ ማ ሜ ም ሞ ሟ
    #         ረ ሩ ሪ ራ ሬ ር ሮ ሯ
    #         ሰ ሱ ሲ ሳ ሴ ስ ሶ ሷ
    #         ሸ ሹ ሺ ሻ ሼ ሽ ሾ ሿ
    #         ቀ ቁ ቂ ቃ ቄ ቅ ቆ ቋ
    #         በ ቡ ቢ ባ ቤ ብ ቦ ቧ
    #         ቨ ቩ ቪ ቫ ቬ ቭ ቮ ቯ
    #         ተ ቱ ቲ ታ ቴ ት ቶ ቷ
    #         ቸ ቹ ቺ ቻ ቼ ች ቾ ቿ
    #         ኋ
    #         ነ ኑ ኒ ና ኔ ን ኖ ኗ
    #         ኘ ኙ ኚ ኛ ኜ ኝ ኞ ኟ
    #         አ ኡ ኢ ኤ እ ኦ
    #         ኧ
    #         ከ ኩ ኪ ካ ኬ ክ ኮ
    #         ኳ
    #         ወ ዉ ዊ ዋ ዌ ው ዎ
    #         ዘ ዙ ዚ ዛ ዜ ዝ ዞ ዟ
    #         ዠ ዡ ዢ ዣ ዤ ዥ ዦ ዧ
    #         የ ዩ ዪ ያ ዬ ይ ዮ
    #         ደ ዱ ዲ ዳ ዴ ድ ዶ ዷ
    #         ጀ ጁ ጂ ጃ ጄ ጅ ጆ ጇ
    #         ገ ጉ ጊ ጋ ጌ ግ ጐ ጓ ጔ
    #         ጠ ጡ ጢ ጣ ጤ ጥ ጦ ጧ
    #         ጨ ጩ ጪ ጫ ጬ ጭ ጮ ጯ
    #         ጰ ጱ ጲ ጳ ጴ ጵ ጶ ጷ
    #         ፀ ፁ ፂ ፃ ፄ ፅ ፆ ፇ
    #         ፈ ፉ ፊ ፋ ፌ ፍ ፎ ፏ
    #         ፐ ፑ ፒ ፓ ፔ ፕ ፖ
    #         """.split()

    #     char_map = {}
    #     char_map["'"] = 0
    #     char_map["UNK"] = 1
    #     char_map["<SPACE>"] = 2
    #     index = 3
    #     for c in supported:
    #         char_map[c] = index
    #         index += 1
    #     index_map = {v+1: k for k, v in char_map.items()}
    #     return char_map, index_map

    def ctc_lambda_func(self, args):
        """CTC lambda function."""
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def add_ctc_loss(self, input_to_softmax):
        """Add a CTC loss to the model."""
        the_labels = tf.keras.Input(
            name='the_labels', shape=(None,), dtype='float32')
        input_lengths = tf.keras.Input(
            name='input_length', shape=(1,), dtype='int64')
        label_lengths = tf.keras.Input(
            name='label_length', shape=(1,), dtype='int64')
        output_lengths = tf.keras.layers.Lambda(
            input_to_softmax.output_length)(input_lengths)
        # CTC loss is implemented in a lambda layer
        loss_out = tf.keras.layers.Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [input_to_softmax.output, the_labels, output_lengths, label_lengths])
        model = Model(
            inputs=[input_to_softmax.input, the_labels,
                    input_lengths, label_lengths],
            outputs=loss_out)
        return model

    def next_train(self, batch_size):
        """ Obtain a batch of training data
        """
        while True:
            ret = self.get_batch('train')
            self.cur_train_index += batch_size
            if self.cur_train_index >= len(self.train_audio_text) - batch_size:
                self.cur_train_index = 0
                self.shuffle_data_by_partition('train')
            yield ret

    def next_valid(self, batch_size):
        """Obtain a batch of validation data."""
        while True:
            ret = self.get_batch('valid')
            self.cur_valid_index += batch_size
            if self.cur_valid_index >= len(self.valid_audio_text) - batch_size:
                self.cur_valid_index = 0
                self.shuffle_data_by_partition('valid')
            yield ret

    def text_to_int_sequence(self, text):
        """ Convert text to an integer sequence """
        char_map, _ = self.loader.map_index()
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = char_map['<SPACE>']
            else:
                # print("checking character " + c + " in map:")
                # print(char_map)
                # check if c exists as a key in char_map
                if c in char_map:
                    ch = char_map[c]
                else:
                    ch = char_map['<UNK>']
                # ch = char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def featurize(self, audio_clip):
        """ For a given audio clip, calculate the corresponding feature
        Params:
            audio_clip (str): Path to the audio clip
        """
        (rate, sig) = wav.read(audio_clip)
        return mfcc(sig, rate, numcep=13)

    def normalize(self, feature, eps=1e-14):
        """ Center a feature using the mean and std
        Params:
            feature (numpy.ndarray): Feature to normalize
        """
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def get_batch(self, partition):
        """ Obtain a batch of train, validation, or test data
        """
        if partition == 'train':
            # audio_paths, text, _ = self.load_transcription(
            #     self.data_dir + '/train/', type='train')
            audio_paths = self.train_audio_paths
            texts = self.train_audio_text
            # audio_paths = self.train_audio_paths
            cur_index = self.cur_train_index
            # texts = text
        elif partition == 'valid':
            # audio_paths, text, _ = self.load_transcription(
            #     self.data_dir + '/test/', type='validation')
            audio_paths = self.valid_audio_paths
            texts = self.valid_audio_text
            # audio_paths = self.valid_audio_paths
            cur_index = self.cur_valid_index
            # texts = self.valid_texts
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

    def train(self, audio_gen, input_to_softmax, model_name, minibatch_size=20, optimizer=tf.keras.optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5), epochs=20, verbose=1):
        """Train the model."""
        self.minibatch_size = minibatch_size
        self.spectrogram = False
        self.fit_train()
        # calculate steps_per_epoch
        # file_path = self.loader.get_file_path()
        # train_audio_paths, _, _ = self.loader.load_transcription(
        #     file_path + 'train/')
        num_train_examples = len(self.train_audio_paths)
        steps_per_epoch = num_train_examples//minibatch_size
        # calculate validation_steps
        # valid_audio_paths, _, _ = self.loader.load_transcription(
        #     file_path + 'test/')
        num_valid_samples = len(self.valid_audio_paths)
        validation_steps = num_valid_samples//minibatch_size

        # add CTC loss to the NN specified in input_to_softmax
        model = self.add_ctc_loss(input_to_softmax)

        # CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
        model.compile(loss={'ctc': lambda y_true,
                      y_pred: y_pred}, optimizer=optimizer)

        # make results/ directory, if necessary
        if not os.path.exists('models'):
            os.makedirs('models')

        # add checkpointer
        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath='models/'+model_name+'.h5', verbose=0)

        # train the model
        hist = model.fit_generator(
            generator=self.next_train(minibatch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.next_valid(minibatch_size),
            validation_steps=validation_steps,
            callbacks=[checkpointer],
            verbose=verbose, use_multiprocessing=True
        )

        # save model loss
        with open('models/'+model_name+'.pickle', 'wb') as f:
            pickle.dump(hist.history, f)

    def cnn_output_length(self, input_length, filter_size, border_mode, stride,
                          dilation=1):
        """ Compute the length of the output sequence after 1D convolution along
            time. Note that this function is in line with the function used in
            Convolution1D class from Keras.
        Params:
            input_length (int): Length of the input sequence.
            filter_size (int): Width of the convolution kernel.
            border_mode (str): Only support `same` or `valid`.
            stride (int): Stride size used in 1D convolution.
            dilation (int)
        """
        if input_length is None:
            return None
        assert border_mode in {'same', 'valid'}
        dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
        if border_mode == 'same':
            output_length = input_length
        elif border_mode == 'valid':
            output_length = input_length - dilated_filter_size + 1
        return (output_length + stride - 1) // stride
