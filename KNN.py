import os
import math
import operator
import csv
import cv2
import numpy as np
from imageDataLoader import ImageDataLoader
from threading import Thread

class KNN:
    def __init__(self, train_dataPath="C:\\Users\\ibrahim mut\\Desktop\\KNN_color_recognition-master\\training_dataset", n_neighbors=5, weights="distance"):

        self.feature_map=[]

        self.dataGenerator=ImageDataLoader(train_dataPath)

    def fit(self, X, verbose=1):

        colors=("b", "g", "r")#cv2 reads as b,g,r

        iteration=0
        for img, belongedClass in X:

            channels=cv2.split(img)

            feature=[]
            for nth_channel, nth_color in zip(channels, colors):

                hist=cv2.calcHist(nth_channel, [0], None, [256], [0,256])

                # find the peak pixel values for R, G, and B
                feature.append( np.argmax(hist) )


            #---VERBOSE PROCESS----------------------
            if verbose>0:
                WIDTH_CONSTANT=70#constant may be max width of terminal
                ratio=int(iteration*WIDTH_CONSTANT/self.dataGenerator.generatorLen)

                #carriage return is may be windows specific
                print(f"\rTraining Process: [",end="")
                print("*"*ratio, end="")
                print(" "*int(WIDTH_CONSTANT-ratio-1), end="")
                print("]", end="")
                iteration+=1
            #---VERBOSE PROCESS----------------------

            with open("model.csv", "a") as f:
                #save found r,g,b feature of belonging class
                f.write(f"{feature[2]}, {feature[1]}, {feature[0]}, {belongedClass}\n")

        print(" Done")

    def predict(self, imageArray, n_neighbors=3):

        channels=cv2.split(imageArray)

        feature=[]
        for nth_channel, nth_color in zip(channels, ("b", "g", "r")):

            hist=cv2.calcHist(nth_channel, [0], None, [256], [0,256])
            # find the peak pixel values for R, G, and B
            feature.append( np.argmax(hist) )


        n=self.findKNearestNeighbors(self.feature_map, feature, n_neighbors)
        scores= self.determineClass(n)
        #print("scores:",scores)
        return max(scores, key= lambda x: x[1])[0]

    def calculateEuclideanDistance(self, array1=[], array2=[], arrayDimensions=0):
        dist=0.0

        for _ in range(arrayDimensions):
            dist+= (array1[_] - array2[_])**2

        return math.sqrt(dist)

    def findKNearestNeighbors(self, known_map, input_sample, n_neighbors=1):


        NUM_THREADS=os.cpu_count()
        map_size=len(known_map)

        threads=[]
        results=[]

        for index in range(0, map_size, NUM_THREADS):

            threadRange=(index, max(index+NUM_THREADS, map_size))

            t=Thread(target=self.__findKNearestNeighborsThread, args=(known_map, input_sample, threadRange, results))
            t.start()
            threads.append( t )

        for thread in threads:
            thread.join()

        results.sort(key=operator.itemgetter(1))#sort by distances
        results= results[:n_neighbors] if 0<n_neighbors<=len(results) else results

        return results

    def __findKNearestNeighborsThread(self, known_map, input_sample, threadRange:tuple=(), result:list=[]):
        dists=[]#distances

        for nth_feature in known_map[threadRange[0]: threadRange[1]]:

            d=self.calculateEuclideanDistance(input_sample, nth_feature, len(input_sample))
            dists.append( (nth_feature, d) )#keep distances to every data points of given input_sample

        dists.sort(key=operator.itemgetter(1))#sort by distances

        #return neighbors
        for neighbor in dists:
            result.append( neighbor )

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
            print("Please provide or train a model. Exiting..")&exit(-1)

        return retval

if __name__=="__main__":
    k=KNN()
    k.fit(k.dataGenerator.generator)#train
    k.loadModel("model.csv")

    img=cv2.imread("C:\\Users\\ibrahim mut\\Desktop\\KNN_color_recognition-master\\training_dataset\\green\\green7.png")
    print("image belongs to class;",k.predict(img, 5))