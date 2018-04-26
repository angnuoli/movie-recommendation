# from preprocess import preprocess
import os

import numpy as np

from adapter import Adapt
from attention import NeuralModel
from postprocess import PostProcess
from preprocess import preprocess
from recommend import CFUtil
from utils.utils import transform

if __name__ == "__main__":
    np.random.seed(100)
    path = os.path.abspath('.')
    pst = PostProcess(path)

    # save data and modify embedding_dim to 300
    samples, users, movies = preprocess()
    samples = samples[0:10000]
    users = transform(users)
    movies = transform(movies)
    # pst.saveSamples(samples, 'samples.csv')
    # pst.saveReviews(users, 'users.csv')
    # pst.saveReviews(movies, 'movies.csv')

    # load data
    # samples = pst.loadSamples('samples.csv')
    # users = pst.loadReviews('users.csv')
    # movies = pst.loadReviews('movies.csv')

    # tuning
    epoch = 10
    l2 = 0.0001
    uhid = 128
    mhid = 128
    nhid = 128
    userMaxLen = 20
    movieMaxLen = 30
    neiMaxLen = 50

    # constant
    sim_thresh = 0.25
    embedding_dim = 300
    userParams = {'user_hid_dim': uhid, 'user_input_length': userMaxLen, 'user_input_dim': embedding_dim}
    movieParams = {'movie_hid_dim': mhid, 'movie_input_length': movieMaxLen, 'movie_input_dim': embedding_dim}
    neiParams = {'nei_hid_dim': nhid, 'nei_input_length': neiMaxLen, 'nei_input_dim': embedding_dim}

    # execute
    cf = CFUtil(samples)
    fullSimilarity = cf.simUser()
    print("Adapt data...")
    adp = Adapt(samples, users, movies, fullSimilarity, userMaxLen, movieMaxLen, neiMaxLen, sim_thresh, embedding_dim)
    User_train_test, Movie_train_test, Neigh_train_test, Y_train_test = adp.kerasInput()
    att = NeuralModel(userParams, movieParams, neiParams, epoch, l2)
    print("Build model...")
    model, history, tesLoss = att.build(User_train_test[0], Movie_train_test[0], Neigh_train_test[0], Y_train_test[0],
                                        User_train_test[1], Movie_train_test[1], Neigh_train_test[1], Y_train_test[1])
    print("Record training history...")
    pst.recordResult(model, history, tesLoss)
