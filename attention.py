# https://github.com/keras-team/keras/issues/4962 for attention model

from keras.layers.pooling import AveragePooling1D
from keras.layers import concatenate, multiply, Activation, Dense, Flatten, Input, Permute, RepeatVector, TimeDistributed, GRU
from keras.models import Model
from keras import regularizers
# from keras.utils import plot_model

import numpy as np
import os
import csv


class Recommend(object):
    def __init__(self, path, userParams, movieParams, epoch, drop, rec_drop, l2):
        self.path = path
        self.user_hid_dim = userParams['user_hid_dim']
        self.user_input_length = userParams['user_input_length']
        self.user_input_dim = userParams['user_input_dim']
        self.movie_hid_dim = movieParams['movie_hid_dim']
        self.movie_input_length = movieParams['movie_input_length']
        self.movie_input_dim = movieParams['movie_input_dim']
        self.epoch = epoch
        self.drop = drop
        self.rec_drop = rec_drop
        self.l2 = l2

    def attentionWrap(self, hid_dim, input_length, input_dim):
        inputs = Input(shape=(input_length, input_dim))
        lstm = GRU(hid_dim, return_sequences=True, dropout=self.drop, recurrent_dropout=self.rec_drop, input_shape=(input_length, input_dim))(inputs)
        att = TimeDistributed(Dense(1, activation='tanh'))(lstm)
        att = Flatten()(att)
        att = Activation(activation='softmax')(att)
        att = RepeatVector(hid_dim)(att)
        att = Permute((2,1))(att)
        mer = multiply([lstm, att])
        hid = AveragePooling1D(pool_size=input_length)(mer)
        hid = Flatten()(hid)
        return inputs, hid

    def build(self, X_user_train, X_movie_train, Y_train, X_user_test, X_movie_test, Y_test):

        input_user, user_rep = self.attentionWrap(self.user_hid_dim, self.user_input_length, self.user_input_dim)
        input_movie, movie_rep = self.attentionWrap(self.movie_hid_dim, self.movie_input_length, self.movie_input_dim)

        x = concatenate([user_rep, movie_rep])
        x = Dense(1, kernel_regularizer=regularizers.l2(self.l2))(x)
        model = Model(inputs=[input_user, input_movie], outputs = x)
        model.compile(loss='mse', optimizer='adam')
        hisObj = model.fit([X_user_train, X_movie_train], Y_train, epochs = self.epoch, validation_split=0.2, verbose =2)
        his = hisObj.history
        loss = model.evaluate([X_user_test, X_movie_test], Y_test, verbose =0)
        print('test loss: ' + str(loss))

        configFILE = os.path.join(self.path,'configJson')
        weightFILE = os.path.join(self.path, 'weight.h5')
        resultFILE = os.path.join(self.path, 'loss.csv')
        picFILE = os.path.join(self.path, 'model.png')
        json_string = model.to_json()
        with open(configFILE, 'w') as f:
            f.write(json_string)
        model.save_weights(weightFILE)

        trainLoss = his['loss']
        valLoss = his['val_loss']
        with open(resultFILE, 'w') as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow([str(x) for x in trainLoss])
            writer.writerow([str(x) for x in valLoss])
            writer.writerow([loss])

        # plot_model(model, to_file=picFILE)

