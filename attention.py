# https://github.com/keras-team/keras/issues/4962 for attention model

from keras.layers.pooling import AveragePooling1D
from keras.layers import concatenate, multiply, Activation, Dense, Flatten, Input, Permute, RepeatVector, TimeDistributed, GRU
from keras.models import Model
from keras import regularizers

import numpy as np

class NeuralModel(object):
    def __init__(self, attParamDic, epoch, l2):
        self.attParamDic = attParamDic
        self.epoch = epoch
        self.l2 = l2

    def attentionWrap(self, hid_dim, input_length, input_dim):
        inputs = Input(shape=(input_length, input_dim))
        lstm = GRU(hid_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, input_shape=(input_length, input_dim))(inputs)
        att = TimeDistributed(Dense(1, activation='tanh'))(lstm)
        att = Flatten()(att)
        att = Activation(activation='softmax')(att)
        att = RepeatVector(hid_dim)(att)
        att = Permute((2,1))(att)
        mer = multiply([lstm, att])
        hid = AveragePooling1D(pool_size=input_length)(mer)
        hid = Flatten()(hid)
        return inputs, hid

    def build(self, name_train_test, Y_train_test, tag):
        mdlNameList = ['user', 'movie']
        if(tag):
            mdlNameList.append('nei')
        fXtrain=[]
        fXtest=[]
        finput = []
        frep = []
        Ytrain = Y_train_test[0]
        Ytest = Y_train_test[1]
        for name in mdlNameList:
            loc_param = self.attParamDic[name]
            loc_input, loc_rep =self.attentionWrap(loc_param[0], loc_param[1], loc_param[2])
            finput.append(loc_input)
            frep.append(loc_rep)
            fXtrain.append(name_train_test[name][0])
            fXtest.append(name_train_test[name][1])
  
        x = concatenate(frep)
        x = Dense(1, kernel_regularizer=regularizers.l2(self.l2))(x)
        model = Model(inputs=finput, outputs = x)
        model.compile(loss='mse', optimizer='adam')
        hisObj = model.fit(fXtrain, Ytrain, epochs = self.epoch, validation_split=0.2, verbose =2)
        loss = model.evaluate(fXtest, Ytest, verbose =0)
        predicts = model.predict(fXtest)

        print('test loss: ' + str(loss))
        return hisObj.history, loss, predicts



