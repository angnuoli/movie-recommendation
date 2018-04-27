import numpy as np

class Adapt(object):
    def __init__(self, samples, users, movies, similarity, userMaxLen, movieMaxLen, neighborMaxLen, threshold, embedding_dim):
        self.samples = samples
        self.embedding_dim = embedding_dim
        self.users = self.rvlist_to_rvnp(users, userMaxLen)
        self.movies = self.rvlist_to_rvnp(movies, movieMaxLen)
        simNeighDic = self.filterUnsimilarUser(similarity, threshold)
        neighs = self.buildGroup(simNeighDic, users, userMaxLen)
        self.neighs = self.rvlist_to_rvnp(neighs, neighborMaxLen)

    def pad(self, reviews, maxLen):
        # reviews = [[], [], [], ...]
        dim = self.embedding_dim
        orgLen = len(reviews)
        if(orgLen < maxLen):
            for i in range(maxLen - orgLen):
                reviews.insert(0, np.zeros(dim))
        else:
            reviews = reviews[orgLen-maxLen : orgLen]
        return reviews

    def rvlist_to_rvnp(self, dic, maxLen):
        newDic = {}
        for key in dic.keys():
            newDic[key] = self.pad(dic[key], maxLen)
        return newDic

    def filterUnsimilarUser(self, similarity, threshold):
        neighborDic = {}
        for user in similarity.keys():
            neighborDic[user] = [simScoreTuple[1] for simScoreTuple in similarity[user] if simScoreTuple[0] > threshold]
        return neighborDic

    def buildGroup(self, neighborDic, users, userMaxLen):
        neighbor_rvs = {}
        for userId in neighborDic:
            reviews = []
            for neiId in neighborDic[userId]:
                neiRvs = users[neiId]
                if userMaxLen < len(neiRvs):
                    neiRvs = neiRvs[len(neiRvs)-userMaxLen : len(neiRvs)]
                for rv in neiRvs:
                    reviews.append(rv)
            neighbor_rvs[userId] = reviews
        return neighbor_rvs

    def splitTrainTest(self, fullData, percent):
        trainLen = int(percent*len(fullData))
        train = fullData[:trainLen]
        test = fullData[trainLen:]
        return train, test

    def kerasInput(self):
        np.random.shuffle(self.samples)
        X_user = []
        X_movie = []
        X_neigh = []
        Y = []

        for smp in self.samples:
            Y.append(smp[2])
            X_user.append(self.users[smp[0]])
            X_movie.append(self.movies[smp[1]])
            X_neigh.append(self.neighs[smp[0]])

        splitPercent = 0.8
        User_train_test = self.splitTrainTest(np.asarray(X_user), splitPercent)
        Movie_train_test = self.splitTrainTest(np.asarray(X_movie), splitPercent)
        Neigh_train_test = self.splitTrainTest(np.asarray(X_neigh), splitPercent)
        Y_train_test = self.splitTrainTest(np.asarray(Y), splitPercent)

        return User_train_test, Movie_train_test, Neigh_train_test, Y_train_test, self.samples