# -*- coding: utf-8 -*-
'''
author:yangyl
'''
from keras import backend as K
import numpy as np
from keras.engine.topology import Layer
from keras import regularizers,initializers
from keras.layers import concatenate
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


class Attention_layer(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # inputs.shape = (batch_size, time_steps, input_dim)

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        super(Attention_layer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):

        # u = tanh(Wx+b)
        uit = K.dot(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)

        gi = K.sum(uit, axis=-1)  # 32 *6
        alfa = K.softmax(gi)
        self.alfa = alfa
        output = x * K.expand_dims(alfa, axis=-1)   # (None, 180, 200)
        # output = K.sum(x * K.expand_dims(alfa, axis=-1), axis=1)  # sum(32 *6 *310)
        print('output..shape', K.int_shape(output))
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[-1])



if __name__ == '__main__':
    from keras.layers import Input,Concatenate,Dense,Dropout, LSTM
    from keras.models import Model
    from keras.layers import Add

    left_input = Input(shape=(5,10), dtype='float32', name='left_input')
    right_input = Input(shape=(5,10), dtype='float32', name='right_input')
    left = LSTM(20, return_sequences=True)(left_input)
    right = LSTM(20, return_sequences=True)(right_input)

    pool =Add()([left,right])
    pool = Attention_layer(name='watt')(pool)

    pool =Dense(10,activation='relu')(pool)
    model =Model(inputs=[left_input,right_input],outputs=pool,name='model12')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    model.summary()