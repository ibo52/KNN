import os
import math
import operator
import csv

class KNN:
    def __init__(self, n_neighbors=5, weights="distance"):
        self.feature_map=[]
        pass

    def fit(self, X, Y):
        pass


    def predict(self, arr, n_neighbors=3):
        n=self.findKNearestNeighbors(self.feature_map, arr, n_neighbors)
        return self.determineClass(n)

    def calculateEuclideanDistance(self, array1=[], array2=[], arrayDimensions=0):
        dist=0.0

        for _ in range(arrayDimensions):
            dist+= (array1[_] - array2[_])**2

        return math.sqrt(dist)

    def findKNearestNeighbors(self, known_map, input_sample, n_neighbors=1):
        dists=[]

        for _ in range(len(known_map)):

            d=self.calculateEuclideanDistance(input_sample, known_map[_], len(input_sample))
            dists.append( (known_map[_], d) )#keep distances to every data points of given input_sample

            dists.sort(key=operator.itemgetter(1))#sort by distances

        #return k nearest neighbors
        if 0 < n_neighbors <= len(dists):
            return dists[:n_neighbors]
        else:
            return dists

    def determineClass(self, neighbors):
        classes={}

        for nth_neighbor in neighbors:

            neighborsClass=nth_neighbor[0][3]# array contains:r,g,b,class, distance to input sample

            if neighborsClass in classes:
                classes[neighborsClass]+=1 #increase the score
            else:
                classes[neighborsClass]=1 #increase the score

        scores = sorted(classes.items(),
                         key=operator.itemgetter(1), reverse=True)
        return scores

    def loadModel(self, path:str):

        retval=0

        if os.path.isfile(path) and os.access(path, os.R_OK):
            print ('KNN Model is loading...')

            with open(path) as csvfile:
                lines = csv.reader(csvfile)
                dataset = list(lines)
                for x in range(len(dataset)):
                    for y in range(3):
                        dataset[x][y] = float(dataset[x][y])
                    self.feature_map.append(dataset[x])
            retval=1
        else:
            print ('No file(or access permissions) for KNN model is found!:',path)

        return retval
k=KNN()
k.loadModel("training.data")
print([255,128,243]," belongs to class;",k.predict([255,128,243], 5))
