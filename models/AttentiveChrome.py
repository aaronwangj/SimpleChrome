from __future__ import absolute_import
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, LSTM, Bidirectional, Permute
from tensorflow.keras import Sequential
from tensorflow.linalg import matmul


def batch_product(input_0, input_1):
    result = None
    for i in range(input_0.shape[0]):
        op = matmul(input_0[i], input_1)
        op = tf.expand_dims(op, 0)
        if result == None:
            result = op
        else:
           result = tf.concat([result, op], axis=0)
    return tf.squeeze(result, axis=2)


class RecurrentAttention(tf.keras.Model):
    """
        Recurrent Attention Module
    """
    def __init__(self, _bin_rnn_size=32, _hm=True, _bidirectional=True):
        super(RecurrentAttention, self).__init__()
        
        if _bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1   
        
        if _hm:
            self.bin_rep_size = _bin_rnn_size * self.num_directions
        else:
            self.bin_rep_size = _bin_rnn_size
        

        self.bin_context_vector = tf.Variable(tf.random.uniform((self.bin_rep_size, 1),
                                    minval=0, 
                                    maxval=None),
                                  trainable=True)
    
    def call(self, _input):
        alpha = tf.nn.softmax(batch_product(_input, self.bin_context_vector ), axis=1)
        batch_size, source_length, _ = _input.shape
        alpha = tf.expand_dims(alpha, 2).reshape(batch_size, -1, source_length)
        return matmul(alpha, _input), alpha



class RecurrentEncoder(tf.keras.Model):
    """
        Recurrent Encoder Module
    """
    def __init__(self, _num_bins, _ip_bin_size, 
                _bin_rnn_size=32,
                _hm=True,
                _bidirectional=True,
                _dropout=0.5):
        super(RecurrentEncoder, self).__init__()
        self.ipshape = _ip_bin_size
        self.seq_length = _num_bins
        self.permute = Permute((2,1,3))
        if _bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        if _hm:
            self.bin_rnn_size = _bin_rnn_size // 2
        else:
            self.bin_rnn_size = _bin_rnn_size

        self.bin_rep_size = self.bin_rnn_size * self.num_directions

        self.rnn = Bidirectional(LSTM(self.bin_rnn_size, dropout=_dropout,
                                      return_sequences=True),
                                      input_shape=((self.ipshape,)))
        
        self.bin_attention = RecurrentAttention()

    def outputlength(self):
        return self.bin_rep_size
    
    def call(self, single_hm, hidden=None):
        bin_output, hidden = self.rnn(single_hm, hidden)
        bin_output = self.permute(bin_output)
        hm_rep, bin_alpha = self.bin_attention(bin_output)
        return hm_rep,bin_alpha


class AttentiveChrome(tf.keras.Model):
    """
        Attentive Chrome Module
    """
    def __init__(self, _num_hms=5,
                _num_bins=200,

        ):
        super(AttentiveChrome, self).__init__()
        self.num_hms = _num_hms
        self.num_bins = _num_bins
        self.ip_bin_size = 1

        self.rnn_hms = []
        self.permute = Permute((2,1,3))

        for i in range(self.num_hms):
            self.rnn_hms.append(RecurrentEncoder(self.num_bins, 
                                                 self.ip_bin_size,
                                                 _hm=False)
                                                )

        self.opsize_0 = self.rnn_hms[0].outputlength()
        self.hm_level_rnn_1 = RecurrentEncoder(self.num_hms,
                                               self.opsize_0,
                                               _hm=True)
        
        self.opsize_1 = self.hm_level_rnn_1.outputlength()
        self.diffopsize = 2*(self.opsize_1)
        self.fdiff1_0 = Dense(1, input_shape=(self.diffopsize,),
                              activation='sigmoid')
    
    def call(self, _input):
        bin_a = None
        level1_rep = None
        [batch_size, _, _] = _input.shape()

        for hm, hm_encdr in enumerate(self.rnn_hms):
            hmod = _input[:,:,hm]
            hmod = tf.expand_dims(tf.transpose(hmod), 2)
            op,a = hm_encdr(hmod)
            if level1_rep is None:
                level1_rep=op
                bin_a=a
            else:
                level1_rep = tf.concat([level1_rep, op], axis=1)
                bin_a = tf.concat([bin_a, a], axis=1)
        
        level1_rep = self.permute(level1_rep)
        final_rep_1, hm_level_attention_1 = self.hm_level_rnn_1(level1_rep)
        final_rep_1 = tf.squeeze(final_rep_1, axis=1)
        prediction_m = self.fdiff1_1(final_rep_1)
        return prediction_m

