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

class AttMemoryLayer(Layer):
    '''
    batch_size =32
       Input: shape[x,memory]
       x is aspect vector with shape(batch_size,1,100)
       memory (batch_size , steps,100)
       Output:
       shape(batch_size , 100)
    '''
    from keras.initializers import he_uniform
    def __init__(self,W_regularizer=None,b_regularizer=None,**kwargs):
        self.supports_masking =False
        # self.mask =mask
        self.W_regularizer =regularizers.get(W_regularizer)
        self.b_regularizer =regularizers.get(b_regularizer)
        super(AttMemoryLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        aspect_shape =input_shape[0]
        m2m_shape =input_shape[1]
        n_in =aspect_shape[2]+m2m_shape[2]
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
        aspect =inputs[0]
        memory =inputs[1]
        print(K.int_shape(aspect))
        aspect =K.reshape(aspect,(-1,K.int_shape(aspect)[2]))
        # (-1,100) ->(-1,n,100)
        vaspect =K.repeat(aspect,K.int_shape(memory)[1])
        print (K.int_shape(aspect))
        print (' vaspect',K.int_shape(vaspect))
        print (' memory',K.int_shape(memory))
        x =concatenate(inputs=[memory,vaspect],axis=-1)
        print ('x...shape',K.int_shape(x))
        gi =K.tanh(K.dot(x,self.W)+self.b)  #32 *6 *1
        gi =K.sum(gi,axis=-1)   # 32 *6
        alfa =K.softmax(gi)
        self.alfa =alfa
        output =K.sum(memory*K.expand_dims(alfa,axis=-1),axis=1) #sum(32 *6 *310)
        print ('output..shape',K.int_shape(output))
        return output
    def compute_output_shape(self, input_shape):
        shape =input_shape[1]
        shape =list(shape)

        return (shape[0],shape[2])
    def compute_mask(self, inputs, mask=None):
        return None

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
        shape =input_shape
        shape =list(shape)

        return  (shape[0],shape[2])

    def compute_mask(self, inputs, mask=None):
        return None


if __name__ == '__main__':
    from keras.layers import Input,Embedding,Highway,Conv1D,MaxPooling1D,Flatten,Concatenate,Dense,Dropout
    from keras.models import Model
    from keras.utils.vis_utils import plot_model
    left_input = Input(shape=(5,10), dtype='float32', name='left_input')

    right_input = Input(shape=(5,10), dtype='float32', name='right_input')
    left =Dense(20,activation='tanh')(left_input)
    right=Dense(20,activation='tanh')(right_input)
    # pool =GatedLayer(name='gate')([left,right])
    # (None, 5, 20)
    pool = AttMemoryLayer(name='watt')([left, right])

    pool =Dense(10,activation='relu')(pool)
    model =Model(inputs=[left_input,right_input],outputs=pool,name='model12')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    model.summary()
    print (model.loss)

    # plot_model(model,'xxx.png',show_shapes=True)
