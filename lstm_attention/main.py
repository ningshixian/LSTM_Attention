#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from __future__ import print_function

'''
    X = Input Sequence of length n.
    H = LSTM(X); Note that here the LSTM has return_sequences = True,
        so H is a sequence of vectors of length n.
    s is the hidden state of the LSTM (h and c)

    h is a weighted sum over H: 加权和
    h = sigma(j = 0 to n-1)  alpha(j) * H(j)

    weight alpha[i, j] for each hj is computed as follows:
    H = [h1,h2,...,hn]
    M = tanh(H)
    alhpa = softmax(w.transpose * M)

    h# = tanh(h)
    y = softmax(W * h# + b)

    J(theta) = negative_log_likelihood + regularity
'''

######################################
# 导入各种用到的模块组件
######################################

# 神经网络的模块
from keras.models import Sequential, Model
from keras.layers import LSTM, Embedding, Lambda, merge, Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.regularizers import l2, activity_l2
from keras.preprocessing import sequence
from keras.datasets import imdb  # 影评数据，被标记为正面/负面两种评价。
from keras import backend as K
# 统计的模块
from collections import Counter
import random, cPickle

# 内存调整的模块
import sys

import numpy as np

np.random.seed(1337)  # for reproducibility

MAX_SEQUENCE_LENGTH = 80  # 每个新闻文本最多保留80个词
MAX_NB_WORDS = 22353  # 字典大小
EMBEDDING_DIM = 100  # 词向量的维度
VALIDATION_SPLIT = 0.2  # 训练集:验证集 = 1:4
BATCH_SIZE = 32
EPOCH = 15  # 迭代次数
LR = 0.01  # 学习率
MOMENTUM = 0.9

######################################
# 加载数据
######################################

print(">> Loading Data ...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb_full.pkl", nb_words=MAX_NB_WORDS)  # (25000L, 1L)
# print(X_train[0])
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('pad_sequences: 将长为nb_samples的序列（标量序列）转化为形如(nb_samples,nb_timesteps)2D numpy array。')
X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
print('\tTherefore, read in', X_train.shape[0], 'samples from the dataset totally.')
print('\tEach sample has ', X_train.shape[1], ' dimension.')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
# label为0~1共2个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
# y_train = np_utils.to_categorical(y_train, 10)


from attention_lstm import AttentionLSTMWrapper, AttentionLSTM

print('Build model...')
model = Sequential()

'''
Embedding层只能作为模型的第一层
输入：最大单词数，即字典长度；句子向量表示的输出维度
# weights=[weights]
'''
weights = np.load('word2vec_100_dim.embeddings')  # (22353L,100L)
model.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=100, weights=[weights]))

'''
outputshape: 如果return_sequences=True，那么输出3维 tensor(nb_samples, timesteps, output_dim) .
否则输出2维tensor(nb_samples,output_dim)。
Exception: Input 0 is incompatible with layer dense_1: expected ndim=2, found ndim=3
'''
# lstm = LSTM(128, W_regularizer=l2(0.01), return_sequences=True)
# model.add(AttentionLSTMWrapper(lstm, single_attention_param=True))
model.add(AttentionLSTM(100, W_regularizer=l2(0.01), dropout_W=0.2, dropout_U=0.2))
model.add(Activation('tanh'))

model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()  # 打印模型的概况


######################################
# 训练LSTM_ATTNets模型
######################################

print('Train...')
print('\tHere, batch_size =', BATCH_SIZE, ", epoch =", EPOCH, ", lr =", LR)
# early_stopping = EarlyStopping(monitor='val_loss', patience=2)
# history可以查看每一次的训练的结果
history = model.fit([X_train], y_train,
                    batch_size=BATCH_SIZE,
                    nb_epoch=EPOCH,
                    verbose=1,
                    validation_split=VALIDATION_SPLIT)
print(history.history)

######################################
# 评估模型
######################################

score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print('Test score:', score)
print('Test accuracy:', acc)


######################################
# 测试模型
######################################

print(">> Test the model ...")
# pre_temp = model.predict_classes(X_test)
predictions = model.predict(X_test)
# print predictions.shape


num = len(predictions)
accuracy = len([1 for i in range(num) if X_test[i] == predictions[i]]) / float(num)
print(">> Report the result ...")
print("\\t0 --> ", '')
print("\\t1 --> ", '')
print("\\tThe accuracy of model is ", acc)
print("\\tThe recall of model is ")
print("\\tThe Fscore of model is ")



######################################
# 保存LSTM_ATTNets模型
######################################

# model.save_weights('MyLSTM_ATTNets.h5')
# cPickle.dump(model, open('./MyLSTM_ATTNets.pkl', "wb"))
# json_string = model.to_json()
# open('LSTM_ATT_Model', 'w').write(json_string)

# 下次要调用这个网络时，用下面的代码
# model = cPickle.load(open(’MyLSTM_ATTNets.pkl',"rb"))
