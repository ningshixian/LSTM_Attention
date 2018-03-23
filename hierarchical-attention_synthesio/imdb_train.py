'''Trains a Hierarchical Attention Model on the IMDB sentiment classification task.
Modified from keras' examples/imbd_lstm.py.
'''
from __future__ import print_function
import numpy as np
from model import createHierarchicalAttentionModel
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.datasets import imdb

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32 

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
#add one extra dimention as the sentence (1 sentence per doc!)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model, modelAttEval = createHierarchicalAttentionModel(maxlen, embeddingSize = 200, vocabSize = max_features)

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
