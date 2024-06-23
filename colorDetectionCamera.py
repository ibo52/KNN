import cv2
from camera import Camera
from KNN import KNN

class ColorDetectionCamera(Camera):
    """
        Opens the default camera, makes prediction if model provided
    """
    def __init__(self, ml_model=None, class_labels:dict=None,model_input_shape:tuple=()):

        super().__init__(ml_model, class_labels, model_input_shape)

    #override
    def run(self):

        while not self.done:
            #get image frame from video device(Threaded)
            self.status, self.frame=self.cap.read()

            crop_end=(128,128)#height, width
            crop_start=( (self.frame.shape[0] - crop_end[0])//2, (self.frame.shape[1] - crop_end[1])//2)#y, x

            rs=self.frame[crop_start[0]:crop_start[0]+crop_end[0] ,
                            crop_start[1]:crop_start[1]+crop_end[1]]

            cv2.rectangle(self.frame,
                          (crop_start[1]-1, crop_start[0]-1 ),
                          (crop_start[1]+crop_end[1], crop_start[0]+crop_end[0]), (64,255,64), 1 )

            #cv2 reads image as b,g,r. reverse the channels
            preds=self.detect(cv2.cvtColor(rs, cv2.COLOR_BGR2RGB), self.model)

            frame_text=f"renk: {preds}"

            cv2.putText(self.frame, frame_text,(40,20),cv2.FONT_HERSHEY_SIMPLEX,1,(32,255,32),2)
            cv2.putText(self.frame, "Rengi bu alana koy",(crop_start[0]+crop_end[0]//2,crop_start[1]-crop_end[1]),cv2.FONT_HERSHEY_SIMPLEX,0.72,(32,255,32),2)

            self.show(self.frame, f"KNN ile histogram frekansı renk tespiti sınaması")
            self.show(rs, f"e renkle ilgili alan")

    #override
    def detect(self, roi, ml_model, n_neighbors=6):
        """Returns the probabilities of classes which image belongs"""

        prediction_results= ml_model.predict(roi, n_neighbors)

        #OPTIONAL: to return the highest possible class that image belongs
        #prediction_results=preds.argmax()

        return prediction_results

if __name__=="__main__":

    model=KNN()
    model.loadModel("tunedModel.csv")#load trained model file

    cam=ColorDetectionCamera(ml_model=model,
               class_labels=None,
               model_input_shape=None)
    cam.run()

