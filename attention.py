# https://github.com/keras-team/keras/issues/4962 for attention model

from keras.layers.pooling import AveragePooling1D
from keras.layers import concatenate, multiply, Activation, Dense, Flatten, Input, Permute, RepeatVector, TimeDistributed
from keras.models import Model
from keras import regularizers
# from keras.utils import plot_model
import numpy as np
import os

np.random.seed(100)

class Recommend(object):
    def __init__(self, path, userParams, movieParams, samples, users, movies):
        self.path = path
        self.user_hid_dim = userParams['user_hid_dim']
        self.user_input_length = userParams['user_input_length']
        self.user_input_dim = userParams['user_input_dim']
        self.movie_hid_dim = moviesParams['movie_hid_dim']
        self.movie_input_length = moviesParams['movie_input_length']
        self.movie_input_dim = moviesParams['movie_input_dim']
        self.samples = samples
        self.users = users
        self.movies = movies

    def transToNumpy(self):
        np.random.shuffle(self.samples)
        X_user = []
        X_movie = []
        Y = []
        for smp in self.samples:
            Y.append(smp[0])

            userList = users[smp[1]]
            user_length = len(userList)

            if(user_length < self.user_input_length):
                for i in range(self.user_input_length - user_length):
                    userList.append(np.ones(self.user_input_dim))
            else:
                userList = userList[len(userList)-self.user_input_length : len(userList)]
            X_user.append(userList)

            movieList = movies[smp[2]]
            movie_length = len(movieList)
            if(movie_length < self.movie_input_length):
                for i in range(self.movie_input_length-movie_length):
                    movieList.append(np.ones(self.movie_input_dim))
            else:
                movieList = movieList[len(movieList - self.movie_input_length) : len(movieList)]
            X_movie.append(movieList)
        return np.asarray(X_user), np.asarray(X_movie), np.asarray(Y)

    def attentionWrap(self, hid_dim, input_length, input_dim):
        inputs = Input(shape=(input_length, input_dim))
        lstm = GRU(hid_dim, dropout=0.2, recurrent_dropout=0.2, input_dim=input_dim, input_length = input_length, return_sequences=True)(inputs)
        att = TimeDistributed(Dense(1, activation='tanh'))(lstm)
        att = Flatten()(att)
        att = Activation(activation='softmax')(att)
        att = RepeatVector(hid_dim)(att)
        att = Permute((2,1))(att)
        mer = multiply([lstm, att])
        hid = AveragePooling1D(pool_length=input_length)(mer)
        hid = Flatten()(hid)
        return inputs, hid

    def build(self):

        # user_hid_dim = 128
        # user_input_length = 50
        # user_input_dim = 300

        # movie_hid_dim = 128
        # movie_input_length = 50
        # movie_input_dim = 300

        input_user, user_rep = self.attentionWrap(self.user_hid_dim, self.user_input_length, self.user_input_dim)
        input_movie, movie_rep = self.attentionWrap(self.movie_hid_dim, self.movie_input_length, self.movie_input_dim)

        x = concatenate([user_rep, movie_rep])
        x = Dense(1, kernel_regularizer=regularizers.l2(0.01))(x)
        model = Model(inputs=[input_user, input_movie], outputs = x)
        model.compile(loss='mse', optimizer='adam')
        X_user, X_movie, Y = transToNumpy()
        trainLen = int(len(X_user)*0.8)
        testLen = len(X_user) - trainLen
        his = model.fit([X_user_train, X_movie_train], Y, verbose =2)
        loss = model.evaluate([X_user_test, X_test], Y_test, verbose =0)
        print("loss: "+ loss)

        # configFILE = os.path.join(path,'configJson')
        # weightFILE = os.path.join(path, 'weight.h5')
        # resultFILE = os.path.join(path, 'result.csv')
        # picFILE = os.path.join(path, 'model.png')
        # json_string = model.to_json()
        # with open(configFILE, 'w') as f:
        #     f.write(json_string)
        # model.save_weights(weightFILE)

        # trainLoss = his['loss']
        # valLoss = his['val_loss']
        # with open(resultFILE, 'wb') as csvfile:
        #     writer=csv.writer(csvfile)
        #     writer.writerow([str(x) for x in trainLoss])
        #     writer.writerow([str(x) for x in valLoss])
        #     writer.writerow([cost])

        # plot_model(model, to_file=picFILE)

