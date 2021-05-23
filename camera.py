import cv2
import numpy as np
import os
import face_recognition
from pkg_resources import resource_exists

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

path = 'BaseImages'
images = []
classNames = []
myList = os.listdir(path)

print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    print(os.path.splitext(cl)[0])
print(classNames)

def findEncoding(images):
    encodeList = []
    for img in images:
        # img = face_recognition.load_image_file(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncoding(images)
print('Encoding Complete, Num: ', len(encodeListKnown))


class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret, frame = self.video.read()
        imgS = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                print(faceLoc)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.rectangle(frame, (x1, y2-35), (x2, y2), (255, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
                cv2.line(frame, (x1, y1), (x1+30, y1), (255, 0, 255), 6)
                cv2.line(frame, (x1, y1), (x1, y1+30), (255, 0, 255), 6)

                cv2.line(frame, (x2, y1), (x2-30, y1), (255, 0, 255), 6)
                cv2.line(frame, (x2, y1), (x2, y1+30), (255, 0, 255), 6)

                cv2.line(frame, (x1, y2), (x1+30, y2), (255, 0, 255), 6)
                cv2.line(frame, (x1, y2), (x1, y2-30), (255, 0, 255), 6)

                cv2.line(frame, (x2, y2), (x2-30, y2), (255, 0, 255), 6)
                cv2.line(frame, (x2, y2), (x2, y2-30), (255, 0, 255), 6)
                
            


        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()