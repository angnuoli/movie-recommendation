import numpy as np

class Adapt(object):
    def __init__(self, samples, users, movies, user_input_length, movie_input_length):
        self.samples = samples
        self.users = users
        self.movies = movies
        self.user_input_length = user_input_length
        self.movie_input_length = movie_input_length

    def transToNumpy(self):
        np.random.shuffle(self.samples)
        X_user = []
        X_movie = []
        Y = []
        for smp in self.samples:
            Y.append(smp[0])

            userList = self.users[smp[1]]
            user_length = len(userList)

            if(user_length < self.user_input_length):
                for i in range(self.user_input_length - user_length):
                    userList.append(np.zeros(self.user_input_dim))
            else:
                userList = userList[len(userList)-self.user_input_length : len(userList)]
            X_user.append(userList)

            movieList = self.movies[smp[2]]
            movie_length = len(movieList)
            if(movie_length < self.movie_input_length):
                for i in range(self.movie_input_length-movie_length):
                    movieList.append(np.zeros(self.movie_input_dim))
            else:
                movieList = movieList[len(movieList) - self.movie_input_length : len(movieList)]
            X_movie.append(movieList)
        return np.asarray(X_user), np.asarray(X_movie), np.asarray(Y)

    def splitTrainTest(self):
        X_user, X_movie, Y = self.transToNumpy()
        trainLen = int(len(X_user)*0.8)
        X_user_train = X_user[:trainLen]
        X_user_test = X_user[trainLen:]
        X_movie_train = X_movie[:trainLen]
        X_movie_test = X_movie[trainLen:]
        Y_train = Y[:trainLen]
        Y_test = Y[trainLen:]
        return X_user_train, X_movie_train, Y_train, X_user_test, X_movie_test, Y_test