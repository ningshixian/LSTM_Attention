# -*- coding: utf-8 -*-
'''
author:yangyl

'''
import os
import cPickle
import numpy as np
from keras.utils.np_utils import to_categorical
from doc_model import HAN
from keras.utils.vis_utils import plot_model
from model_utils import train_val_split,fbeta_score
from model_utils import LossHistory,plot_loss,ResultHistory

DATAPATH ='./BioData/Doc_data.pkl'
SAVEPATH='./Results/HAN-ATT'

if os.path.isdir(SAVEPATH):
    pass
else:
    os.makedirs(SAVEPATH)
train_data,test_data,train_label,test_label,vecs =cPickle.load(open(DATAPATH,'rb'))

train_label =to_categorical(train_label,2)
test_label=to_categorical(test_label,2)
train_data =np.array(train_data,dtype='int32')
test_data =np.array(test_data,dtype='int32')

np.random.seed(2017)
max_features =vecs.shape[0]
embedding_dim =vecs.shape[1]
MAX_SENT_LENGTH=100
MAX_SENTS=15
model =HAN(MAX_SENT_LENGTH,MAX_SENTS,max_features,embedding_dim,vecs)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[fbeta_score,'acc'])

print("model fitting - Bidirectional LSTM")
model.summary()
plot_model(model,SAVEPATH+'/model.png',show_shapes=True)
x_train, y_train, x_val, y_val =train_val_split(train_data,train_label,2017,0.2)
loss_his = LossHistory()
result_his =ResultHistory(test_data,SAVEPATH,False)
result_dev_his =ResultHistory(x_val,SAVEPATH,False)
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=128, callbacks=[loss_his,result_his,result_dev_his])
plot_loss(SAVEPATH, loss_his)
i =0
for result in result_his.result:
    i += 1
    np.savetxt(SAVEPATH + '/result_' + str(i) + '.txt', result, fmt="%.4f", delimiter=" ")
i=0
for result in result_dev_his.result:
    i += 1
    np.savetxt(SAVEPATH + '/dev_result_' + str(i) + '.txt', result, fmt="%.4f", delimiter=" ")
np.savetxt(SAVEPATH+'/dev_label.txt',y_val,fmt="%i",delimiter=' ')

if __name__ == '__main__':
    pass