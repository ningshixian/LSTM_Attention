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
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM
from keras.regularizers import l2
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from attention_lstm import AttentionLSTM_t, AttentionLSTMWrapper
from keras.datasets import imdb

np.random.seed(1337)  # for reproducibility


def preditFval(predictions, test_label):
    num = len(predictions)
    with open('L_predict_result.txt', 'w') as f:
        for i in range(num):
            if predictions[i][1] > predictions[i][0]:
                predict = +1
            else:
                predict = -1
            f.write(str(predictions[i][0]) + ' ' + str(predictions[i][1]) + '\n')

    TP = len([1 for i in range(num) if
              predictions[i][1] > predictions[i][0] and (test_label[i] == np.asarray([0, 1])).all()])
    FP = len([1 for i in range(num) if
              predictions[i][1] > predictions[i][0] and (test_label[i] == np.asarray([1, 0])).all()])
    FN = len([1 for i in range(num) if
              predictions[i][1] < predictions[i][0] and (test_label[i] == np.asarray([0, 1])).all()])
    TN = len([1 for i in range(num) if
              predictions[i][1] < predictions[i][0] and (test_label[i] == np.asarray([1, 0])).all()])

    print('Wether match? ', (TP + FP + FN + TN) == num)
    print(TP, FP, FN, TN)  # 0 0 1875 9803

    precision = TP / (float)(TP + FP)
    recall = TP / (float)(TP + FN)
    Fscore = (2 * precision * recall) / (precision + recall)  # ZeroDivisionError: integer division or modulo by zero

    print(">> Report the result ...")
    print("-1 --> ", len([1 for i in range(num) if predictions[i][1] < predictions[i][0]]))
    print("+1 --> ", len([1 for i in range(num) if predictions[i][1] > predictions[i][0]]))
    print("TP=", TP, "  FP=", FP, " FN=", FN, " TN=", TN)
    print("precision= ", precision)
    print("recall= ", recall)
    print("Fscore= ", Fscore)


def buildLstmAtt():

    print('Build model...')
    model = Sequential()

    model.add(Embedding(input_dim=max_features,
                        output_dim=128,
                        # weights=[embedding_matrix],
                        trainable=True,
                        input_length=maxlen))

    # lstm = LSTM(100, W_regularizer=l2(0.01))
    # model.add(AttentionLSTMWrapper(lstm, single_attention_param=True))
    model.add(AttentionLSTM_t(128, W_regularizer=l2(0.01), dropout_W=0.2, dropout_U=0.2))

    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


if __name__ == '__main__':

    max_features = 20000
    # cut texts after this number of words
    # (among top max_features most common words)
    maxlen = 100
    batch_size = 32

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    y_train = np.array(y_train)
    y_test = np.array(y_test)


    model = buildLstmAtt()
    model.summary()  # 打印模型的概况

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=4,
              validation_data=[x_test, y_test])

    model.save('attention_lstm_model.h5')
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    predictions = model.predict(x_test)


######################################
# 保存LSTM_ATTNets模型
######################################

# model.save_weights('MyLSTM_ATTNets.h5')
# cPickle.dump(model, open('./MyLSTM_ATTNets.pkl', "wb"))

# json_string = model.to_json()
# open('LSTM_ATT_Model', 'w').write(json_string)

# 下次要调用这个网络时，用下面的代码
# model = cPickle.load(open(’MyLSTM_ATTNets.pkl',"rb"))
