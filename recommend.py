from math import *

class CFUtil(object):
    def __init__(self, samples):
        self.critics = self.criDic(samples)

    def criDic(self, samples):
        cridic = {}
        for smp in samples:
            userId = smp[0]
            movieId = smp[1]
            rate = smp[2]
            exist = userId in cridic
            if not exist:
                cridic[userId] = {}
            cridic[userId][movieId] = rate
        return cridic

    def pearsion(self, prefs, p1, p2):
        # Get the list of mutually rated items
        si={}
        for item in prefs[p1]: 
            if item in prefs[p2]:
                si[item]=1

        # if they are no ratings in common, return 0
        if len(si)==0: return 0

        # Sum calculations
        n=len(si)

        # Sums of all the preferences
        sum1=sum([prefs[p1][it] for it in si])
        sum2=sum([prefs[p2][it] for it in si])

        # Sums of the squares
        sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
        sum2Sq=sum([pow(prefs[p2][it],2) for it in si])   

        # Sum of the products
        pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])

        # Calculate r (Pearson score)
        num=pSum-(sum1*sum2/n)
        den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
        if den==0: return 0

        r=num/den
        return r

    def topMatches(self, prefs, person):
        scores=[(self.pearsion(prefs,person,other),other) 
                      for other in prefs if other!=person]
        scores.sort()
        return scores

    def simUser(self):
        simDic = {}
        for key in self.critics:
            simDic[key] = self.topMatches(self.critics, key)
        return simDic
