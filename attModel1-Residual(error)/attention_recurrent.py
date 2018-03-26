'''
https://github.com/ningshixian/LSTM_Attention/blob/master/attModel1/custom_recurrents.py

ValueError: None values not supported.
'''
import tensorflow as tf
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec
from utils.tdd import _time_distributed_dense

tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)

class AttentionDecoder(Recurrent):

    def __init__(self, units, output_dim,
                 score_func,
                 activation='tanh',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Implements an AttentionDecoder that takes in a sequence encoded by an
        encoder and outputs the decoded states
        :param units: dimension of the hidden state and the attention matrices
        :param output_dim: the number of labels in the output space
        references:
            Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
            "Neural machine translation by jointly learning to align and translate."
            arXiv preprint arXiv:1409.0473 (2014).
        """
        self.units = units
        self.output_dim = output_dim
        self.score_func = score_func
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences
        self.eps = 1e-5

    def build(self, input_shape):
        """
          See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
          for model details that correspond to the matrices here.
        """

        self.batch_size, self.timesteps, self.input_dim = input_shape

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        self.states = [None, None]  # y, s

        """
            Matrices for creating the context vector
        """

        # self.V_A_h = self.add_weight(shape=(self.units, ),
        #                            name='V_A_h',
        #                            initializer=self.kernel_initializer,
        #                            regularizer=self.kernel_regularizer,
        #                            constraint=self.kernel_constraint)

        self.W_A_X = self.add_weight(shape=(self.units, self.units),
                                   name='W_A_X',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_A_h = self.add_weight(shape=(self.units, self.units),
                                   name='W_A_h',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_A_combine = self.add_weight(shape=(self.units*2, self.units),
                                     name='W_A_combine',
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
        self.b_A_X = self.add_weight(shape=(self.units,),
                                   name='b_A_X',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        self.b_A_combine = self.add_weight(shape=(self.units,),
                                     name='b_A_combine',
                                     initializer=self.bias_initializer,
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)

        # for attention weight and score function
        if self.score_func == "Euclidean":
            self.W_A = self.add_weight(shape=(self.units, ),
                                               name='W_A',
                                                initializer='one',
                                               regularizer=self.kernel_regularizer,
                                               constraint=self.kernel_constraint)
            self.scoreFun = self.euclideanScore
        elif self.score_func == "Manhatten":
            self.scoreFun = self.manhattenScore
            self.W_A = self.add_weight(shape=(self.units, ),
                                       name='W_A',
                                       initializer='one',
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint)

        else:
            assert 0, "we only have Euclidean, Bilinear, forwardNN" \
                      " score function for attention"

        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True

    def call(self, x):
        # store the whole sequence so we can "attend" to it at each timestep
        self.x_seq = x

        # apply the a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:
        self._uxpb = _time_distributed_dense(self.x_seq, self.W_A_X, b=self.b_A_X,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self.units)

        return super(AttentionDecoder, self).call(x)


    def get_initial_state(self, inputs):
        print('inputs shape:', inputs.get_shape())

        # apply the matrix on the first time step to get the initial s0.
        s0 = inputs
        return [s0, 0]


    def step(self, x, state):

        all, t = state
        ht = all[:, t]  # (None, 128)
        _stm = K.repeat(ht, self.timesteps) # (None, 100, 128)
        _Wxstm = K.dot(_stm, self.W_A_h)

        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.

        # et = K.dot(activations.tanh(_Wxstm + self._uxpb),
        #            K.expand_dims(self.W_A))
        et = self.scoreFun(_Wxstm, self._uxpb, K.expand_dims(self.W_A))

        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated  # vector of size # (None, 100, 1)

        # calculate the context vector
        context_g = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)  # (None, 128)
        zt = K.tanh(K.dot(K.concatenate([context_g, ht]), self.W_A_combine))    # (None, 128)
        return zt, [all, t+1]


    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def euclideanScore(self, attended, state, W):
        # {{{
        # Euclidean distance
        M = (attended - state) ** 2
        M = K.dot(M, W)
        # energy = M.max() - M
        return M

    def manhattenScore(self, attended, state, W):
        # {{{
        # Manhattan Distance
        # eps for avoid gradient to be NaN;
        M = K.abs(K.maximum(attended - state, self.eps))
        M = K.dot(M, W)
        _energy = M.max() - M
        return _energy



# check to see if it compiles
if __name__ == '__main__':
    from keras.layers import Input, LSTM,TimeDistributed, Dense, Flatten
    from keras.models import Model
    from keras.layers.wrappers import Bidirectional
    import numpy as np

    i = Input(shape=(100,100))
    enc = Bidirectional(LSTM(64, return_sequences=True), merge_mode='concat')(i)    # (None, 100,128)
    dec = AttentionDecoder(128, 100, score_func='Euclidean')(enc)
    dec = Flatten()(dec)
    re = Dense(1,activation='sigmoid')(dec)
    model = Model(inputs=i, outputs=re)
    model.summary()

    n = np.random.uniform(-0.1, 0.1, 100*100)
    n = n.reshape(1, 100, 100)  # reshape array to be 3x5
    label = np.ones(1)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    model.fit(n,label,batch_size=32, epochs=1)
