# from preprocess import preprocess
from attention import NeuralModel
from adapter import Adapt
from recommend import CFUtil
from postprocess import PostProcess

import numpy as np
import os

if __name__ == "__main__":
    np.random.seed(100)
    path = os.path.abspath('.')
    pst = PostProcess(path)

    # save data and modify embedding_dim to 300
    # samples, users, movies = preprocess()
    samples, users, movies = preprocess()
    # pst.saveSamples(samples, 'samples.csv')
    # pst.saveReviews(users, 'users.csv')
    # pst.saveReviews(movies, 'movies.csv')

    # load data
    # samples = pst.loadSamples('samples.csv')
    # users = pst.loadReviews('users.csv')
    # movies = pst.loadReviews('movies.csv')

    # tuning
    epoch=1
    l2 = 0.01
    uhid = 128
    mhid = 128
    nhid = 128
    userMaxLen = 20
    movieMaxLen = 40
    neiMaxLen = 60
    usingNeiModel = False
    jobName = 'default'

    # constant
    sim_thresh = 0.5
    embedding_dim = 300
    attParamDic = {'user': [uhid, userMaxLen, embedding_dim], 
    'movie': [mhid, movieMaxLen, embedding_dim], 'nei': [nhid, neiMaxLen, embedding_dim]}
    path = os.path.abspath('.')
    pst = PostProcess(path)

    #excute
    cf = CFUtil(samples)
    fullSimilarity = cf.simUser()
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