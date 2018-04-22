import numpy as np
import os
from attention import Recommend
from adapter import Adapt

np.random.seed(100)

epoch=1
l2 = 0.01
uhid = 128
mhid = 128
userMaxLen = 50
movieMaxLen = 50 

# mustn't modify the following parameters
path = os.path.abspath('.')
path = os.path.join(path, 'result')
drop = 0.2
rec_drop = 0.2
userParams = {'user_hid_dim': uhid, 'user_input_length': userMaxLen, 'user_input_dim': 300}
movieParams = {'movie_hid_dim': mhid, 'movie_input_length': movieMaxLen, 'movie_input_dim': 300}

users = {}
for i in range(20):
    users[i] = [np.random.rand(300) for j in range(50)]
movies = {}
for j in range(30):
    movies[j] = [np.random.rand(300) for j in range(50)]
samples = []
for i in range(20):
    for j in range(30):
        samples.append(( float(np.random.randint(0, 5)), i, j))

adp = Adapt(samples, users, movies, userMaxLen, movieMaxLen)
X_user_train, X_movie_train, Y_train, X_user_test, X_movie_test, Y_test = adp.splitTrainTest()


att = Recommend(path, userParams, movieParams, epoch, drop, rec_drop, l2)
att.build(X_user_train, X_movie_train, Y_train, X_user_test, X_movie_test, Y_test)