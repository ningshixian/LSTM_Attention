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
from keras.layers import Embedding
from keras.layers.core import Dense
from keras.regularizers import l2
import numpy as np
import cPickle
from process_data import make_idx_data_cv
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from attention_lstm import AttentionLSTM_t

np.random.seed(1337)  # for reproducibility


def loadData(path):
    x = cPickle.load(open(path, "rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print(len(word_idx_map))
    print(len(vocab))
    datasets = make_idx_data_cv(revs, word_idx_map, 1, max_l=10, k=100, filter_h=1)
    img_h = len(datasets[0][0]) - 1
    test_set_x = datasets[1][:, :img_h]
    test_set_y = np.asarray(datasets[1][:, -1], "int32")
    train_set_x = datasets[0][:, :img_h]
    train_set_y = np.asarray(datasets[0][:, -1], "int32")
    print(np.shape(train_set_x))
    print('load data...')
    print(np.shape(W))
    print(type(W))
    return (train_set_x, train_set_y), (test_set_x, test_set_y), W


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


if __name__ == '__main__':
    MAX_SEQUENCE_LENGTH = 10  # 每个新闻文本最多保留80个词
    MAX_NB_WORDS = 9870L
    # MAX_NB_WORDS = 22353  # 字典大小
    EMBEDDING_DIM = 100  # 词向量的维度
    VALIDATION_SPLIT = 0.2  # 训练集:验证集 = 1:4
    BATCH_SIZE = 128
    EPOCH = 1  # 迭代次数
    LR = 0.01  # 学习率
    MOMENTUM = 0.9

    # print('Loading data...')
    # (x_train, y_train), (x_test, y_test), embedding_matrix, MAX_NB_WORDS = preprocecss.process()

    print('Loading data...')
    path = '../data/corpus/mr_Lscope.p'
    (x_train, y_train), (x_test, y_test), embedding_matrix = loadData(path)

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    def buildLstmAtt():
        '''构建模型'''
        print('Build model...')
        model = Sequential()

        model.add(Embedding(input_dim=MAX_NB_WORDS,
                            output_dim=EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            # trainable=False,
                            input_length=MAX_SEQUENCE_LENGTH))

        # lstm = LSTM(100, W_regularizer=l2(0.01))
        # model.add(AttentionLSTMWrapper(lstm, single_attention_param=True))
        model.add(AttentionLSTM_t(100, W_regularizer=l2(0.01), dropout_W=0.2, dropout_U=0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        return model

    model = buildLstmAtt()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['accuracy'])
    model.summary()  # 打印模型的概况

    print('Train...')
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCH, validation_split=VALIDATION_SPLIT)
    model.save('attention_lstm_model.h5')
    score, acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print('Test score:', score)
    print('Test accuracy:', acc)

    predictions = model.predict(x_test)
    preditFval(predictions, y_test)


######################################
# 保存LSTM_ATTNets模型
######################################

# model.save_weights('MyLSTM_ATTNets.h5')
# cPickle.dump(model, open('./MyLSTM_ATTNets.pkl', "wb"))

# json_string = model.to_json()
# open('LSTM_ATT_Model', 'w').write(json_string)

# 下次要调用这个网络时，用下面的代码
# model = cPickle.load(open(’MyLSTM_ATTNets.pkl',"rb"))
