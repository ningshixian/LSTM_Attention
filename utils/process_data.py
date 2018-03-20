import numpy as np
import pickle as pkl
from collections import defaultdict
import sys, re
import pandas as pd
import os

"""
Load data --mr_Lscope.p
"""

def build_data_cv(data_folder, clean_string=True):
    """
    Loads data
    """
    revs = []
    train_context_file = data_folder[0]
    train_label_file = data_folder[1]
    test_context_file = data_folder[2]
    test_label_file = data_folder[3]

    trainTag = 0
    testTag = 1

    posTag = "+1"
    negPos = "-1"

    vocab = defaultdict(float)
    with open(train_context_file, "r") as f:
        train_label = open(train_label_file, "r")
        for line in f:
            label = train_label.readline().strip();
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            #print(orig_rev)##############
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            polarity = 0
            if label == posTag:
                polarity = 1;
            else:
                polarity = 0;
            datum  = {"y":polarity,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": trainTag}
                      # "split": np.random.randint(0,cv)}
            revs.append(datum)
        train_label.close()
    with open(test_context_file, "r") as f:
        test_label = open(test_label_file, "r")
        for line in f:
            label = test_label.readline().strip();
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            #print(orig_rev)##############
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            polarity = 0
            if label == posTag:
                polarity = 1;
            else:
                polarity = 0;
            datum  = {"y":polarity,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": testTag}
                      # "split": np.random.randint(0,cv)}
            revs.append(datum)
        test_label.close()

    return revs, vocab

def get_W(word_vecs, k=100,path='wordemb'):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    f =open(path+'.word','w')
    for word in word_vecs:
        # print("#############################")
        # print(word_vecs[word].shape)
        # print(word)
        # print(W[i].shape)
        # print("#############################")
        W[i] = word_vecs[word]
        f.write(word+'\n')
        word_idx_map[word] = i
        i += 1
    np.savetxt(path+'.txt',W,fmt='%.7f', delimiter=' ')
    f.close()
    return W, word_idx_map

def load_vec(fname, vocab):
    """
    format: word vec[50]
    """
    word_vecs = {}
    #print(vocab)
    with open(fname, "r") as f:
        for line in f:
            strs =line.strip().split(' ')
            if strs[0] in vocab:

                word_vecs[strs[0]] = np.array([float(elem) for elem in strs[1:]], dtype='float32')

    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=100):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            # print("************************")
            # print(word)
            # print("************************")
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        if rev["split"]==cv:
            test.append(sent)
        else:
            train.append(sent)
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]

import json

if __name__=="__main__":
    cnnJson = open("test.json", "r")
    inputInfo = json.load(cnnJson)
    cnnJson.close()

    TraiContextFile = inputInfo["TraiContext"]
    TestContextFile = inputInfo["TestContext"]
    TraiLabelFile = inputInfo["TraiLabel"]
    TestLabelFile = inputInfo["TestLabel"]
    wordVectorFile = inputInfo["WordVector"]
    # outputPath = inputInfo["OutPutPath"]
    mrPath = inputInfo["mrPath"]
    saveEMPath = inputInfo["saveEmpath"]
    k = 100
    if not os.path.exists(saveEMPath):
        os.makedirs(saveEMPath)
    w2v_file = wordVectorFile
    data_folder = [TraiContextFile, TraiLabelFile, TestContextFile, TestLabelFile]
    print("loading data...")
    revs, vocab = build_data_cv(data_folder,clean_string=False)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    print("loading word2vec vectors...",)
    w2v = load_vec(w2v_file, vocab)
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab, k = k)
    W, word_idx_map = get_W(w2v, k,saveEMPath+'/'+str(k)+'wordvec')
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, k = k)
    W2, _ = get_W(rand_vecs, k,saveEMPath+'/'+str(k)+'random')
    pkl.dump([revs, W, W2, word_idx_map, vocab], open(mrPath, "wb"))
    print("dataset created!")

