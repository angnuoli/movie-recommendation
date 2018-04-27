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
usingNeiModel = False

# mustn't modify the following parameters

sim_thresh = 0.5
embedding_dim = 5
attParamDic = {'user': [uhid, userMaxLen, embedding_dim], 
'movie': [mhid, movieMaxLen, embedding_dim], 'nei': [nhid, neiMaxLen, embedding_dim]}
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
User_train_test, Movie_train_test, Neigh_train_test, Y_train_test, sflSmps = adp.kerasInput()
X_train_test= {'user':User_train_test, 'movie':Movie_train_test, 'nei':Neigh_train_test}
att = NeuralModel(attParamDic, epoch, l2)
history, tesLoss, predicts = att.build(X_train_test, Y_train_test, usingNeiModel)
comparison = []
for i in range(len(predicts)):
    smp = sflSmps[i]
    comparison.append([smp[0], smp[1], smp[2], predicts[i][0]])

pst.recordResult(history, tesLoss, comparison, fileModifier='l2_e-2')

