import cv2
import argparse
#import time
import os
import eel
import light

frequency=2500
duration=1000

eel.init('WD')
emotions=["angry", "happy", "sad", "neutral"]
ff = cv2.face.FisherFaceRecognizer_create()
font = cv2.FONT_HERSHEY_SIMPLEX
'''try:
    fishface.load("model.xml")
except:
    print("No trained model found... --update will create one.")'''

parser=argparse.ArgumentParser(description="Options for emotions based music player(Updating the model)")
parser.add_argument("--update", help="Call for taking new images and retraining the model.", action="store_true")
args=parser.parse_args()    
facedict={}
video_capture=cv2.VideoCapture(0)
facecascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def crop(clahe_image, face):
    for (x, y, w, h) in face:
        faceslice=clahe_image[y:y+h, x:x+w]
        faceslice=cv2.resize(faceslice, (48, 48))
        facedict["face%s" %(len(facedict)+1)]=faceslice
    return faceslice

def grab_face():
    #ret, frame=video_capture.read()
    ret, frame=light.nolight()
    #cv2.imshow("Video", frame)
    cv2.imwrite('test.jpg', frame)
    cv2.imwrite("images/main%s.jpg" %count, frame)
    gray=cv2.imread('test.jpg',0)
    #gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image=clahe.apply(gray)
    return clahe_image

def detect_face():
    clahe_image=grab_face()
    face=facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face)>=1:
        faceslice=crop(clahe_image, face)
        #return faceslice
    else:
        print("No/Multiple faces detected!!, passing over the frame")



def get_emotion():
    prediction=[]
    confidence=[]

    for i in facedict.keys():
        pred, conf=ff.predict(facedict[i])
        cv2.imwrite("images/%s.jpg" %i, facedict[i])
        prediction.append(pred)
        confidence.append(conf)
    output=emotions[max(set(prediction), key=prediction.count)]    
    print("You seem to be %s" %output) 
    facedict.clear()
    return output;
    #songlist=[]
    #songlist=sorted(glob.glob("songs/%s/*" %output))
    #random.shuffle(songlist)
    #os.startfile(songlist[0])
count=0
@eel.expose
def getEmotion():
   
    count=0
    while True:
        count=count+1
        detect_face()
        if args.update:
            #update_model(emotions)
            break
        elif count==10:
            ff.read("model3.xml")
            return get_emotion()
            break

eel.start('main.html')
    
