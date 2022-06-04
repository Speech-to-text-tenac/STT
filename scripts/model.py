"""Model for STT engine."""
import sys

import tensorflow as tf
from logger import Logger
from tensorflow.keras.models import Model


class STTModel():
    def __init__(self):
        """Initialize STT modeling."""
        try:
            self.logger = Logger("model.log").get_app_logger()
            self.logger.info(
                'Successfully Instantiated STTModel Class Object')
        except Exception as e:
            self.logger.error(
                'Failed to Instantiate STTModel Class Object')
            self.logger.error(e)
            sys.exit(1)

    def model(self, input_dim, filters, kernel_size, conv_stride,
              conv_border_mode, units, output_dim=29, dropout_rate=0.3, number_of_layers=2,
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
        model.output_length = lambda x: cnn_output_length(
            x, kernel_size, conv_border_mode, conv_stride)
        print(model.summary())
        # plot_model(model, to_file='models/model.png', show_shapes=True)
        return model

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
        char_map["UNK"] = 1
        char_map["<SPACE>"] = 2
        index = 3
        for c in supported:
            char_map[c] = index
            index += 1
        index_map = {v+1: k for k, v in char_map.items()}
        return char_map, index_map

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
