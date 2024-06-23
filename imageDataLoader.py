import os
import cv2
import numpy as np

class ImageDataLoader:
    def __init__(self, path, rotate=0, brightness=None):

        self.labels={}
        self.generatorLen=0

        self.__beforeLoadData(path)
        self.generator=self.__augmentedData(self.__loadData(path))#generator object to yield images from train path

    def __beforeLoadData(self, path):
        #extracting class labels
        if os.path.isdir(path):
            dirs=os.listdir(path)

            c=0
            for subPaths in dirs:

                classDir=os.path.join(path,subPaths)

                if  os.path.isdir( classDir ):
                    self.labels[c]=subPaths
                    c+=1

                #find total count of all images belongs to classes
                for img in os.listdir( classDir ):

                    imgPath=os.path.join(classDir, img)

                    if os.path.isfile(imgPath):
                        self.generatorLen+=1

            print(f"[{self.__class__.__name__}]: Found {self.generatorLen} images belonging to {c} classes")

    def __augmentedData(self, imageGenerator, brightness=None):
        for img in imageGenerator:

            if brightness==True:

                B,G,R=cv2.split(img)

                B = cv2.equalizeHist(B)
                G = cv2.equalizeHist(G)
                R = cv2.equalizeHist(R)

                yield cv2.merge((B,G,R))

            yield img



    def __loadData(self, path):
        """yields a data generator for given path"""

        if os.path.isdir(path):
            dirs=os.listdir(path)

            for subPaths in dirs:
                classDir=os.path.join(path,subPaths)

                if  os.path.isdir( classDir ):

                    #feeding data with yield to be memory efficient
                    for img in os.listdir( classDir ):

                        imgPath=os.path.join(classDir, img)

                        if os.path.isfile(imgPath):
                            #return X, Y
                            yield (cv2.imread(imgPath), subPaths)#here cv2 stores image as np array

if __name__=="__main__":
    k=ImageDataLoader("C:\\Users\\ibrahim mut\\Desktop\\KNN_color_recognition-master\\training_dataset")
    print(k.labels)