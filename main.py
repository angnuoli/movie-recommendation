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

    samples, users, movies = preprocess()
    samples = samples[0:20000]
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
    l2 = 0.01
    uhid = 128
    mhid = 128
    nhid = 128
    userMaxLen = 5
    movieMaxLen = 10
    neiMaxLen = 15
    usingNeiModel = True
    jobName = 'default'

    # constant
    sim_thresh = 0.25
    embedding_dim = 300
    attParamDic = {'user': [uhid, userMaxLen, embedding_dim], 
    'movie': [mhid, movieMaxLen, embedding_dim], 'nei': [nhid, neiMaxLen, embedding_dim]}
    path = os.path.abspath('.')
    pst = PostProcess(path)

    # execute
    cf = CFUtil(samples)
    fullSimilarity = cf.simUser()
    print("Adapt data...")
    adp = Adapt(samples, users, movies, fullSimilarity, userMaxLen, movieMaxLen, neiMaxLen, sim_thresh, embedding_dim)

    User_train_test, Movie_train_test, Neigh_train_test, Y_train_test, sflSmps = adp.kerasInput()
    X_train_test= {'user':User_train_test, 'movie':Movie_train_test, 'nei':Neigh_train_test}
    att = NeuralModel(attParamDic, epoch, l2)
    history, tesLoss, predicts = att.build(X_train_test, Y_train_test, usingNeiModel)
    comparison = []
    for i in range(len(predicts)):
        smp = sflSmps[i]
        comparison.append([smp[0], smp[1], smp[2], predicts[i][0]])

    pst.recordResult(history, tesLoss, comparison, fileModifier=jobName)
