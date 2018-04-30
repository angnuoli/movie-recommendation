# from keras.utils import plot_model
import csv
import os


class PostProcess(object):
    def __init__(self, path):
        self.path = path

    def saveSamples(self, samples, file):
        workSpace = os.path.join(self.path, 'data')
        filePath = os.path.join(workSpace, file)
        with open(filePath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for userId_movieId_rate in samples:
                writer.writerow([str(x) for x in userId_movieId_rate])

    def saveReviews(self, id_reviews_dic, file):
        workSpace = os.path.join(self.path, 'data')
        filePath = os.path.join(workSpace, file)
        with open(filePath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([str(len(id_reviews_dic))])
            for key in id_reviews_dic.keys():
                writer.writerow([str(key)])
                reviews = id_reviews_dic[key]
                writer.writerow([str(len(reviews))])
                for rv in reviews:
                    writer.writerow([str(x) for x in rv])

    def loadReviews(self, file):
        workSpace = os.path.join(self.path, 'data')
        filePath = os.path.join(workSpace, file)
        id_reviews = {}
        with open(filePath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            userLen = int(next(reader)[0])
            for i in range(userLen):
                ID = next(reader)[0]
                rvLen = int(next(reader)[0])
                rvs = []
                for j in range(rvLen):
                    rv = next(reader)
                    rvs.append([float(x) for x in rv])
                id_reviews[ID] = rvs
        return id_reviews

    def loadSamples(self, file):
        workSpace = os.path.join(self.path, 'data')
        filePath = os.path.join(workSpace, file)
        samples = []
        with open(filePath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                smp = []
                smp.append(row[0])
                smp.append(row[1])
                smp.append(float(row[2]))
                samples.append(smp)
        return samples

    def recordResult(self, history, testLoss, trgt_prdt, fileModifier = 'default'):
        workSpace = os.path.join(self.path, 'result')
        # configFILE = os.path.join(workSpace,'configJson')
        # weightFILE = os.path.join(workSpace, 'weight.h5')
        lossFILE = os.path.join(workSpace, fileModifier+'_loss.csv')
        cmpFILE = os.path.join(workSpace, fileModifier+'_rateCmp.csv')
        # picFILE = os.path.join(workSpace, 'model.png')
        # json_string = model.to_json()
        # with open(configFILE, 'w') as f:
        #     f.write(json_string)
        # model.save_weights(weightFILE)

        trainLoss = history['loss']
        valLoss = history['val_loss']
        with open(lossFILE, 'w', newline='') as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow([str(x) for x in trainLoss])
            writer.writerow([str(x) for x in valLoss])
            writer.writerow([testLoss])

        with open(cmpFILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for cmp in trgt_prdt:
                writer.writerow([str(x) for x in cmp]) 
