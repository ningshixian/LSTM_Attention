# -*- coding: utf-8 -*-
'''
author:yangyl

'''
import keras.callbacks as callbacks
import  keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.engine import Layer
import keras.regularizers as regularizers
class ResultHistory(callbacks.Callback):
    def __init__(self, data,save_path,is_saving=False):
        self.data = data
        self.save_path =save_path
        self.is_saving =is_saving
        super(ResultHistory, self).__init__()

    def on_train_begin(self, logs={}):
        self.result = []

    def on_epoch_end(self, epoch, logs=None):
        X_test = self.data
        result = self.model.predict(X_test)
        if self.is_saving:
            self.model.save_weights(self.save_path+'/weights_'+str(epoch)+'.hdf5')
        print('\n')

        self.result.append(result)

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_loss = []
        self.epoch = []
        self.fbeta =[]
        self.val_fbeta =[]
    def on_epoch_end(self, epoch, logs=None):
        self.epoch.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.fbeta.append(logs.get('fbeta_score'))
        self.val_fbeta.append(logs.get('val_fbeta_score'))

class AttentivePoolingLayer(Layer):

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
        print 'memory shape',K.int_shape(memory)
        gi =K.tanh(K.dot(memory,self.W)+self.b)  #32 *6 *1
        gi =K.sum(gi,axis=-1)   # 32 *6
        alfa =K.softmax(gi)
        self.alfa =alfa
        output =K.sum(memory*K.expand_dims(alfa,axis=-1),axis=1) #sum(32 *6 *310)
        print 'output..shape',K.int_shape(output)
        return output
    def compute_output_shape(self, input_shape):
        shape =input_shape
        shape =list(shape)

        return  (shape[0],shape[2])

    def compute_mask(self, inputs, mask=None):
        return None

def precision(y_true, y_pred):
    print ('label shape:',K.int_shape(y_true))
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true,y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true,y_pred):
    p =precision(y_true,y_pred)
    r =recall(y_true,y_pred)
    bb =1
    fbeta_score =(1+bb)*(p*r)/(bb*p +r+K.epsilon())
    return fbeta_score

def train_val_split(data,labels,seed,size):
    np.random.seed(seed)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(size * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    return x_train,y_train,x_val,y_val
def plot_loss(save_path,history):
    plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    plt.sca(ax1)
    plt.plot(history.epoch, history.losses, '-or', label="loss")
    plt.plot(history.epoch, history.val_loss, '-xb', label="val_loss")
    plt.legend(loc='upper right')
    plt.sca(ax2)
    plt.plot(history.epoch, history.fbeta, '-or', label="fbeta_score")
    plt.plot(history.epoch, history.val_fbeta, '-xb', label="val_fbeta_score")
    plt.legend(loc='upper right')
    plt.savefig(save_path+'/MyLoss.jpg')
    plt.show()

def calcuate_doc_len():
    TRAINPATH = './BioData/doc/CDR_TrainingSet.data.nn'
    DEVPATH = './BioData/doc/CDR_DevelopmentSet.data.nn'
    TESTPATH = './BioData/doc/CDR_TestSet.data.nn'
    train = map(lambda x: x.strip(), open(TRAINPATH, 'r'))
    dev = map(lambda x: x.strip(), open(DEVPATH, 'r'))
    test = map(lambda x: x.strip(), open(TESTPATH, 'r'))
    doc_has_sent =[]
    doc_len =[]
    sent_has_word =[]

    # calcuate the max length of doc, sentence and word
    for data in (train, dev, test):
        for i in data:

            doc_has_sent.append(len(i.split('. ')))

            doc_len.append(len(i.split(' ')))

            for w in i.split('. '):
                sent_has_word.append(len(w.split(' ')))


    doc_len =pd.Series(doc_len)
    doc_has_sent =pd.Series(doc_has_sent)
    sent_has_word=pd.Series(sent_has_word)
    print doc_has_sent.describe()
    print doc_len.describe()
    print sent_has_word.describe()

if __name__ == '__main__':

    calcuate_doc_len()