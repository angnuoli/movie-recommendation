import numpy as np
import os
from attention import NeuralModel
from adapter import Adapt
from recommend import CFUtil

np.random.seed(100)

epoch=1
l2 = 0.01
uhid = 128
mhid = 128
nhid = 128
userMaxLen = 50
movieMaxLen = 50
neiMaxLen = 50

# mustn't modify the following parameters
path = os.path.abspath('.')
path = os.path.join(path, 'result')
sim_thresh = 0.5
embedding_dim = 5
userParams = {'user_hid_dim': uhid, 'user_input_length': userMaxLen, 'user_input_dim': embedding_dim}
movieParams = {'movie_hid_dim': mhid, 'movie_input_length': movieMaxLen, 'movie_input_dim': embedding_dim}
neiParams = {'nei_hid_dim': nhid, 'nei_input_length': neiMaxLen, 'nei_input_dim': embedding_dim}

users = {}
for i in range(6):
    users[i] = [np.random.rand(embedding_dim) for k in range(30)]

movies = {}
for j in range(10):
    movies[j] = [np.random.rand(embedding_dim) for k in range(40)]

samples = []
for i in range(6):
    for j in range(10):
        samples.append((i, j, float(np.random.randint(0, 5))))

cf = CFUtil(samples)
fullSimilarity = cf.simUser()
adp = Adapt(samples, users, movies, fullSimilarity, userMaxLen, movieMaxLen, neiMaxLen, sim_thresh, embedding_dim)
User_train_test, Movie_train_test, Neigh_train_test, Y_train_test = adp.kerasInput()
att = NeuralModel(path, userParams, movieParams, neiParams, epoch, l2)
att.build(User_train_test[0], Movie_train_test[0], Neigh_train_test[0], Y_train_test[0], User_train_test[1], Movie_train_test[1], Neigh_train_test[1], Y_train_test[1])