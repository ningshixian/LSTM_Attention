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
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)

        a = K.exp(uit)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

            # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # a = K.expand_dims(a)
        weighted_input = x * a
        print('output..shape', K.int_shape(weighted_input))
        # weighted_input = K.sum(weighted_input, axis=1)
        return weighted_input

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[-1])


class AttentivePoolingLayer(Layer):
    '''
    比起AttMemoryLayer，这个层没有拼接关系R
    '''
    from keras.initializers import he_uniform
    def __init__(self,W_regularizer=None,b_regularizer=None,**kwargs):
        self.supports_masking =False
        # self.mask =mask
        self.W_regularizer =regularizers.get(W_regularizer)
        self.b_regularizer =regularizers.get(b_regularizer)
        super(AttentivePoolingLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        # m2m_shape =input_shape[0]
        n_in =input_shape[2]
        n_out =1
        lim =np.sqrt(6./(n_in+n_out))
        # tanh initializer xavier
        self.W =K.random_uniform_variable((n_in,n_out),-lim,lim,
                                         name='{}_W'.format(self.name) )
        self.b =K.zeros((n_out,),name='{}_b'.format(self.name))
        self.trainable_weights=[self.W,self.b]
        self.regularizer =[]
        if self.W_regularizer is not None:
            self.add_loss(self.W_regularizer(self.W))
        if self.b_regularizer is not None:
            self.add_loss(self.b_regularizer(self.b))
        self.build =True
    def call(self, inputs,mask=None):
        # memory =inputs[0]
        memory =inputs
        print ('memory shape',K.int_shape(memory))
        gi =K.tanh(K.dot(memory,self.W)+self.b)  #32 *6 *1
        gi =K.sum(gi,axis=-1)   # 32 *6
        alfa =K.softmax(gi)
        self.alfa =alfa
        output =K.sum(memory*K.expand_dims(alfa,axis=-1),axis=1) #sum(32 *6 *310)
        print ('output..shape',K.int_shape(output))
        return output
    def compute_output_shape(self, input_shape):
        # shape =input_shape[0]
        shape =input_shape
        shape =list(shape)

        return  (shape[0], shape[2])

    def compute_mask(self, inputs, mask=None):
        return None


if __name__ == '__main__':
    from keras.layers import Input,Concatenate,Dense,Dropout, LSTM
    from keras.models import Model
    from keras.utils.vis_utils import plot_model
    left_input = Input(shape=(5,10), dtype='float32', name='left_input')

    right_input = Input(shape=(5,10), dtype='float32', name='right_input')
    # left =Dense(20,activation='tanh')(left_input)
    # right=Dense(20,activation='tanh')(right_input)
    left = LSTM(20, return_sequences=True)(left_input)
    right = LSTM(20, return_sequences=True)(right_input)
    # pool =GatedLayer(name='gate')([left,right])
    from keras.layers import Add
    pool =Add()([left,right])
    pool = Attention_layer(name='watt')(pool)

    pool =Dense(10,activation='relu')(pool)
    model =Model(inputs=[left_input,right_input],outputs=pool,name='model12')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    model.summary()
    print (model.loss)