import numpy as np
import os
from attention import NeuralModel
from adapter import Adapt
from recommend import CFUtil
from postprocess import PostProcess

def save(postpro, samples, users, movies):
    postpro.saveSamples(samples, 'samples.csv')
    postpro.saveReviews(users, 'users.csv')
    postpro.saveReviews(movies, 'movies.csv')
    
def load(postpro):
    samples = postpro.loadSamples('samples.csv')
    users = postpro.loadReviews('users.csv')
    movies = postpro.loadReviews('movies.csv')
    return samples, users, movies

np.random.seed(100)

epoch=1
l2 = 0.01
uhid = 128
mhid = 128
nhid = 128
userMaxLen = 20
movieMaxLen = 40
neiMaxLen = 60

# mustn't modify the following parameters

sim_thresh = 0.5
embedding_dim = 5
userParams = {'user_hid_dim': uhid, 'user_input_length': userMaxLen, 'user_input_dim': embedding_dim}
movieParams = {'movie_hid_dim': mhid, 'movie_input_length': movieMaxLen, 'movie_input_dim': embedding_dim}
neiParams = {'nei_hid_dim': nhid, 'nei_input_length': neiMaxLen, 'nei_input_dim': embedding_dim}
path = os.path.abspath('.')
pst = PostProcess(path)

# users = {}
# for i in range(10):
#     users[i] = [np.random.rand(embedding_dim) for k in range(20)]

# movies = {}
# for j in range(15):
#     movies[j] = [np.random.rand(embedding_dim) for k in range(40)]

# samples = []
# for i in range(10):
#     for j in range(15):
#         samples.append((i, j, float(np.random.randint(0, 5))))
# save(pst, samples, users, movies)

samples, users, movies = load(pst)

cf = CFUtil(samples)
fullSimilarity = cf.simUser()
adp = Adapt(samples, users, movies, fullSimilarity, userMaxLen, movieMaxLen, neiMaxLen, sim_thresh, embedding_dim)
User_train_test, Movie_train_test, Neigh_train_test, Y_train_test = adp.kerasInput()
att = NeuralModel(userParams, movieParams, neiParams, epoch, l2)
model, history, tesLoss = att.build(User_train_test[0], Movie_train_test[0], Neigh_train_test[0], Y_train_test[0], User_train_test[1], Movie_train_test[1], Neigh_train_test[1], Y_train_test[1])

pst.recordResult(model, history, tesLoss)

