#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
在Keras模型中使用预训练的词向量
本文所使用的 W2V 词向量是在搜狗新闻上训练的，有？？个不同的词，每个词用300维向量表示。

以下是我们如何解决分类问题的步骤：

1、将所有的训练样本转化为词索引序列。所谓词索引就是为每一个词依次分配一个整数ID。
2、生成一个词向量矩阵。第i列表示词索引为i的词的词向量。
3、将词向量矩阵载入Keras Embedding层，设置该层的权重不可再训练（也就是说在之后的网络训练过程中，词向量不再改变）。
4、最后用一个softmax全连接输出序列的标注
'''

from __future__ import print_function
import codecs
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from io import open  # 同时兼容2和3的open函数
from tqdm import tqdm

def readEmbedFile(embFile, dim):
    """
    读取预训练的词向量文件，引入外部知识
    :param embFile:
    :return:
    """
    print("\nProcessing Embedding File...")
    embeddings = {}
    embeddings["PADDING_TOKEN"] = np.zeros(dim)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.25, 0.25, dim)
    embeddings["NUMBER"] = np.random.uniform(-0.25, 0.25, dim)

    with codecs.open(embFile, encoding='utf-8') as f:
        lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        values = line.strip().split()
        word = values[0]
        vector = np.asarray(map(float, values[1:]), dtype=np.float32)
        embeddings[word] = vector

    print('Found %s word vectors.' % len(embeddings))  # 400000
    return embeddings


def get_word_index(trainFile, testFile):
    """
    1、建立索引字典- word_index
    :param trainFile:
    :param testFile:
    :return:
    """
    print('\n获取索引字典- word_index \n')
    max_len = 0
    word_counts = {}
    # 一次性读入文件，注意内存
    with codecs.open(trainFile, 'r', 'utf-8') as trainf:
        with codecs.open(testFile, 'r', 'utf-8') as testf:
            lines = trainf.readlines()
            lines.extend(testf.readlines())
    for line in lines:
        if line in ['\n', '\r\n']:
            continue
        seq = line.strip().split()  # 空格分隔，以词组作为基本词
        if len(seq) > max_len:
            max_len = len(seq)
        for w in seq:
            if w in word_counts:
                word_counts[w] += 1
            else:
                word_counts[w] = 1
    # 根据词频来确定每个词的索引
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts]
    # note that index 0 is reserved, never assigned to an existing word
    word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

    # 加入未登陆新词和填充词
    word_index['retain-padding'] = 0
    word_index['retain-unknown'] = len(word_index)
    print('word_index 长度：%d' % len(word_index))
    print('max_len 长度：%d' % max_len)
    return word_index, max_len


def sent2vec2(sent, word_index):
    """
    将一行转换为转化为词索引序列
    :param sent:
    :param word_index:
    :return:
    """
    charVec = []
    for char in sent:
        if char in word_index:
            charVec.append(word_index[char])
        else:
            charVec.append(word_index['retain-unknown'])
        # return charVec[:]
        return [i for i in charVec if i]


def doc2vec(ftrain, ftrain_label, ftest, ftest_label, word_index):
    """
    2、将全部训练和测试语料 转化为词索引序列
    :param ftrain:
    :param ftrain_label:
    :param ftest:
    :param ftest_label:
    :param word_index:
    :return:
    """
    x_train, y_train, x_test, y_test = [], [], [], []
    tags = ['-1', '+1']  # 标注统计信息对应 [ 1.  0.] [ 0.  1.]

    # 读入训练文件，注意内存
    with codecs.open(ftrain, 'r', 'utf-8') as fd:
        lines = fd.readlines()
        for line in lines:
            if line not in ['\n', '\r\n']:
                line = line.replace('\n', '')
                split_line = line.split()
                index_line = sent2vec2(split_line, word_index)
                x_train.append(index_line)
        with codecs.open(ftrain_label, 'r', 'GBK') as label:
            for line in label:
                flag = line.strip()
                index = tags.index(flag)  # 标签 label 记得转换成数组下标！！
                y_train.append(index)

        # 读入测试文件，注意内存
        with codecs.open(ftest, 'r', 'utf-8') as fd:
            lines = fd.readlines()
        for line in lines:
            if line not in ['\n', '\r\n']:
                line = line.replace('\n', '')
                split_line = line.split()
                line = sent2vec2(split_line, word_index)
                x_test.append(line)
        with codecs.open(ftest_label, 'r', 'GBK') as label:
            for line in label:
                flag = line.strip()
                index = tags.index(flag)  # 标签 label 记得转换成数组下标！！
                y_test.append(index)

        return x_train, y_train, x_test, y_test


def process_data(data_label, word_index, max_len):
    """
    3、将转化后的 词索引序列 转化为神经网络训练所用的张量
    :param data_label:
    :param word_index:
    :param max_len:
    :return:
    """
    ltrain = data_label[0]
    label1 = data_label[1]
    ltest = data_label[2]
    label2 = data_label[3]
    x_train, y_train, x_test, y_test = doc2vec(ltrain, label1, ltest, label2, word_index)

    # 填充序列至固定长度，在每个样本前补 0
    x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
    x_test = pad_sequences(x_test, maxlen=max_len, padding='post')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)
    return (x_train, y_train), (x_test, y_test)


def get_w(word_index, embeddings_index, EMBEDDING_DIM):
    """
    4、生成一个词向量矩阵
    :param word_index:
    :param embeddings_index:
    :param EMBEDDING_DIM:
    :return:
    """
    print('\nPreparing embedding matrix......\n')
    MAX_NB_WORDS = 200000  # 字典大小
    nb_words = min(MAX_NB_WORDS, len(word_index))

    # embedding_matrix = 2 * np.random.random((nb_words, EMBEDDING_DIM)) - 1
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i == 0: continue
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index.get(word)
        elif word.lower() in embeddings_index:
            embedding_matrix[i] = embeddings_index.get(word.lower())
        else:
            vec = np.random.uniform(-1, 1, size=EMBEDDING_DIM)  # 随机初始化
            embedding_matrix[i] = vec
    return embedding_matrix, nb_words


if __name__ == '__main__':
    EMBEDDING_DIM = 110  # (100 for word2vec embeddings and 10 for extra features )
    num_classes = 2
    EMBEDDING_FILE = 'embedding/word2vec_100_dim.txt'  # wordphrase100_pos_10.bin	word2vec_100_dim.txt

    BASE = 'data/phrase_cue/'
    FTRAIN_DATA_DIR = BASE + 'Ftrain.txt'
    FTEST_DATA_DIR = BASE + 'Ftest.txt'
    LTRAIN_DATA_DIR = BASE + 'Ltrain.txt'
    LTEST_DATA_DIR = BASE + 'Ltest.txt'

    FTRAIN_LABEL_DIR = 'label/Ftrainlabel.txt'
    FTEST_LABEL_DIR = 'label/Ftestlabel.txt'
    LTRAIN_LABEL_DIR = 'label/Ltrainlabel.txt'
    LTEST_LABEL_DIR = 'label/Ltestlabel.txt'

    data_label = [LTRAIN_DATA_DIR, LTRAIN_LABEL_DIR, LTEST_DATA_DIR, LTEST_LABEL_DIR]
    # data_label = [FTRAIN_DATA_DIR, FTRAIN_LABEL_DIR, FTEST_DATA_DIR, FTEST_LABEL_DIR]

    embeddings_index = readEmbedFile(EMBEDDING_FILE, EMBEDDING_DIM)
    word_index, max_len = get_word_index(data_label[0], data_label[2])
    (x_train, y_train), (x_test, y_test) = process_data(data_label, word_index, max_len)
    embedding_matrix, num_words = get_w(word_index, embeddings_index, EMBEDDING_DIM)

