# -*- coding: utf-8 -*-
'''
author:yangyl

'''
from keras.layers import Dense, Input,LSTM,Conv1D,GlobalMaxPool1D,Masking
from keras.layers import  Embedding, Dropout,Bidirectional,Concatenate,TimeDistributed
from keras.models import Model
from model_utils import AttentivePoolingLayer

def HAN(MAX_SENT_LENGTH,MAX_SENTS,max_features,embedding_dim,vecs):
    embedding_layer = Embedding(max_features,
                                embedding_dim,
                                weights=[vecs],
                                input_length=MAX_SENT_LENGTH,
                                trainable=True,

                                )
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    # embedded_sequences =Masking()(embedded_sequences)
    l_lstm = Bidirectional(LSTM(150,return_sequences=True))(embedded_sequences)
    l_lstm =AttentivePoolingLayer()(l_lstm)
    sentEncoder = Model(sentence_input, l_lstm)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    # review_encoder =Masking()(review_encoder)
    l_lstm_sent = Bidirectional(LSTM(50,return_sequences=True))(review_encoder)
    l_lstm_sent =AttentivePoolingLayer()(l_lstm_sent)
    preds = Dense(2, activation='softmax')(l_lstm_sent)
    model = Model(review_input, preds)
    return model
