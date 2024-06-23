from keras.models import Sequential, load_model
from threading import Thread
import cv2
import os
import numpy as np

class Camera:
    """
        Opens the default camera, makes prediction if model provided
    """
    def __init__(self, ml_model:Sequential=None, class_labels:dict=None,model_input_shape:tuple=()):

        #machine learning model to use on camera
        self.model=ml_model
        self.labels=class_labels
        self.model_input_shape=model_input_shape

        #open default video device from system
        self.cap=cv2.VideoCapture(0)

        #OPTIONAL:main frame to manipulate and show contents
        #we can add labels, add multiple images etc.
        self.canvas=None
        self.frame=np.zeros( (32,32,3) )#camera frame returned from read()
        self.status=True#camera status returned from read()

        self.done=False

    def run(self):

        #TODO: pre processing and variable declarations to use on next steps
        #EXAMPLE: canvas=np.zeros( (640, 480, 3) )#rgb canvas to lay the image and put labels on

        #threading the image capture process to avoid blocking on single thread
        #self.captureThread=Thread(target=self.capture, args=())
        #self.captureThread.daemon=True
        #self.captureThread.start()

        while not self.done:
            #get image frame from video device(Threaded)
            self.status, self.frame=self.cap.read()

            #TODO: pre processing of the image in purpose of giving it to the model
                    #EXAMPLE: imgOfInterest=cv2.resize(captured_image,
            	#(desired_resolution_x, desired_resolution_y),interpolation=cv2.INTER_AREA)#resize to marker model input size

            #TODO: return detection results from model
            #EXAMPLE: prediction_results=self.detectLandmarks(imgOfInterest, your_ML_model)

            #TODO: model prediction: image preparation
            rs=cv2.resize(self.frame,(64,64) , cv2.INTER_AREA)
            #rs=cv2.cvtColor(rs, cv2.COLOR_BGR2GRAY)
            #self.frame=self.frame/255.0
            rs=rs/255.0
            preds=self.detect(rs, self.model)
            #print("-"*40)
            #TODO: model detection
            results= self.detect(rs, self.model)#imageOfInterest, ml_model)
            out=preds.argmax()

            frame_text=f"{labels_[out]}:"+"{:.2f}".format(preds[0][out])

            #TODO: OPTIONAL apply filter or put text to frame according to detected class
            #formattedText=f"{preds[preds.argmax]}%:{labels_[preds.argmax()]}"
            cv2.putText(self.frame, frame_text,(40,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

            self.show(self.frame, f"some window text")

    def capture(self):
        """
            Private function
            captures the image from device and store in its structure
"""
        while self.cap.isOpened():
            self.status, self.frame=self.cap.read()

    def detect(self, roi, ml_model):
        """Returns the probabilities of classes which image belongs"""

        prediction_results= ml_model.predict(np.reshape(roi,(-1,self.model_input_shape[0],
                                                             self.model_input_shape[1],
                                                             self.model_input_shape[2])
                                                        ), verbose=0
                                             )

        #OPTIONAL: to return the highest possible class that image belongs
        #prediction_results=preds.argmax()

        return prediction_results

    def show(self, frame:np.array, window_text:str="Camera window"):
        """
            Shows the given image in graphical interface
        """
        cv2.imshow(f"{window_text}",frame)#cv2.resize(frame,(0,0),fx=2,fy=2))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            self.done=True
            exit(0)


if __name__=="__main__":
    print("""How to use that module:
1. load your trained model
2. load class labels
3. initialize a Camera and pass them as parameters
EXAMPLE:
       from keras.models import load_model
       from labels.py import labels_

       classifier = load_model('model/yourModelFile.h5')
       cam=Camera(classifier, labels_)
       cam.run()
""")


    from models.renkModel.properties import labels_, INPUT_SHAPE

    temp=dict()

    for item in labels_.items():
        temp[item[1]]=item[0]

    labels_=temp
    del temp

    cam=Camera(ml_model=load_model(
        os.path.join(os.getcwd(),"models","renkModel/renkModel.keras")),
               class_labels=labels_,
               model_input_shape=INPUT_SHAPE)
    cam.run()

