from preprocess import preprocess
from attention import NeuralModel
from adapter import Adapt
from recommend import CFUtil

import numpy as np
import os

if __name__ == "__main__":
    np.random.seed(100)
    samples, users, movies = preprocess()

    epoch=10
    l2 = 0.01
    uhid = 128
    mhid = 128
    nhid = 128
    userMaxLen = 20
    movieMaxLen = 30
    neiMaxLen = 50

    # mustn't modify the following parameters
    path = os.path.abspath('.')
    path = os.path.join(path, 'result')
    sim_thresh = 0.25
    embedding_dim = 300
    userParams = {'user_hid_dim': uhid, 'user_input_length': userMaxLen, 'user_input_dim': embedding_dim}
    movieParams = {'movie_hid_dim': mhid, 'movie_input_length': movieMaxLen, 'movie_input_dim': embedding_dim}
    neiParams = {'nei_hid_dim': nhid, 'nei_input_length': neiMaxLen, 'nei_input_dim': embedding_dim}

    # excute
    cf = CFUtil(samples)
    fullSimilarity = cf.simUser()
    adp = Adapt(samples, users, movies, fullSimilarity, userMaxLen, movieMaxLen, neiMaxLen, sim_thresh, embedding_dim)
    User_train_test, Movie_train_test, Neigh_train_test, Y_train_test = adp.kerasInput()
    att = NeuralModel(path, userParams, movieParams, neiParams, epoch, l2)
    att.build(User_train_test[0], Movie_train_test[0], Neigh_train_test[0], Y_train_test[0], User_train_test[1], Movie_train_test[1], Neigh_train_test[1], Y_train_test[1])
