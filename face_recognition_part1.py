import cv2
import numpy as np

face_classifier =cv2.cascadeClassifier('C:/Users/Nishit/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/haar-cascade-files-master.zip/haarcascade-frontalface_default')

def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiscale(gray,1.3,5)
    
    if face is():
       return None

    for(x,y,w,h) in faces:
       cropped_face=img[y:y+h,x:x+w]
    
    return cropped_face     





cap=cv2.VideoCapture(0)
count=0


while True:
    ret,frame=cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
           
        file_name_path='C:/Users/Nishit/Documents/faces/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('face cropper',face)
    else:
        print("face not found")
        pass
    if cv2.waitKey(1)==13 or count ==100:
        break

cap.release()
cv2.destroyAllWindows()
print("collecting samples complete")      
    