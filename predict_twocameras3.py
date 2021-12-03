from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
#import tensorflowjs as tfjs
import cv2  
import pygame.mixer
import time

import sqlite3
con = sqlite3.connect('database.db')
cur = con.cursor()

import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


classes = ({0:'angry',1:'disgust',2:'fear',3:'happy',
        4:'sad',5:'surprise',6:'neutral'})

face_cascade_file = './haarcascade_frontalface_default.xml'  
front_face_detector = cv2.CascadeClassifier(face_cascade_file)  

model_path = './trained_models/fer2013_mini_XCEPTION.110-0.65.hdf5'
emotions_XCEPTION = load_model(model_path, compile=False)


i = 0
flag = True
captures = []

WIDTH = 320
HEIGHT = 240

WIDTH = 640
HEIGHT = 480

def func2():
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    global cam1_judge_face

    while True:
        cur.execute('SELECT max(id), angry,disgust,fear,happy,sad,surprise,neutral FROM cam1;')
        out = [float(i) for i in cur.fetchall()[0][1:]]
        cur.execute('SELECT max(id), angry,disgust,fear,happy,sad,surprise,neutral FROM cam2;')
        out2 = [float(i) for i in cur.fetchall()[0][1:]]
        
        #print(out)
        #con.commit()
        #cam1_judge_face=0

        if out.index(max(out))==3 and out[3]>=0.8:
            cam1_judge_face=1
            break
            #print('笑ったね！！！！')
        elif out2.index(max(out2))==3 and out2[3]>=0.8:
            cam1_judge_face=2
            break
        
            
        #print(type(out[0]))
        

    # try:
    #     lock.acquire(True)
    #     print(1)
    #     cur.execute('SELECT max(id), angry,disgust,fear,happy,sad,surprise,neutral FROM cam1;')
    #     out = cur.fetchall()[0]
    #     print(out)
    #     print(2)
    # finally:
    #     print(3)
    #     lock.release()
    # print(out)
    #     cun+=1
    #     if cun==100:
    #         break

def func1():
    print(0)
    pygame.mixer.init()
    pygame.mixer.music.load("./bgm/Countdown06-2.mp3")
    pygame.mixer.music.play(1)
    time.sleep(6)
    pygame.mixer.music.stop()
    #executor.submit(func2)
    func2()
    print(1)

    # con = sqlite3.connect('database.db')
    # cur = con.cursor()


    # cur.execute('SELECT * FROM cam1;')
    # print(2)
    # out = cur.fetchall()[0]
    # print(out)
    # print(3)
    # con.commit()

for i in [1,3]:
    capture = cv2.VideoCapture(i)
    ret, frame = capture.read()
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    minW = 0.2*capture.get(cv2.CAP_PROP_FRAME_WIDTH)  
    minH = 0.2*capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  
    captures.append( capture )
    #flag = ret

#cur = con.cursor()
cam1_judge_face=0

alpha = 0.5
beta = 1 - alpha

while(True):
    for i, capture in enumerate( captures ):
        ret, img = capture.read()

        tick = cv2.getTickCount()  

        #ret, img = cap.read()
        #print(img)
        if(ret == False):
            print('ret false')
            continue
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = front_face_detector.detectMultiScale(   
            gray,  
            scaleFactor = 1.2,  
            minNeighbors = 3,  
            minSize = (int(minW), int(minH)),  
           )  

        for(x,y,w,h) in faces:  

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)  
            face_cut = img[y:y+h, x:x+w]
            face_cut = cv2.resize(face_cut,(64, 64))
            face_cut = cv2.cvtColor(face_cut, cv2.COLOR_BGR2GRAY)

            img_array = image.img_to_array(face_cut)
            pImg = np.expand_dims(img_array, axis=0) / 255

            prediction = emotions_XCEPTION.predict(pImg)[0]
            #print(prediction)
            #print(prediction.argsort())
            
            top_indices = prediction.argsort()[-5:][::-1]
            #print(top_indices)

            result = sorted([[classes[i], prediction[i]] for i in top_indices], reverse=True, key=lambda x
: x[1])
            cur.execute("INSERT INTO cam%d (angry,disgust,fear,happy,sad,surprise,neutral) VALUES (%f,%f,%f,%f,%f,%f,%f)"%(i+1, prediction[0], prediction[1], prediction[2], prediction[3], prediction[4], prediction[5], prediction[6]))
            con.commit()

            #print(result)
            result_c = result[0][0]

            cv2.putText(img, str(result_c), (x+5,y-5), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), 2)

        if cam1_judge_face==1 and i==0:
            cv2.putText(img, "www",
                        (50, 300), cv2.FONT_HERSHEY_PLAIN, 15, (0, 0, 255), 5, cv2.LINE_AA)
        elif cam1_judge_face==2 and i==1:
            cv2.putText(img, "www",
                        (50, 300), cv2.FONT_HERSHEY_PLAIN, 15, (0, 0, 255), 5, cv2.LINE_AA)

            # overlay = frame.copy()
            # img = frame.copy()

            # cv2.rectangle(overlay, (0, 0), (640, 480), (0, 0, 255), -1)
            # cv2.addWeighted(overlay, alpha, img, beta, 0, img)

            
        # FPS算出と表示用テキスト作成  
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)  
        # FPS  
        cv2.putText(img, "FPS:{} ".format(int(fps)),   
            (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2, cv2.LINE_AA)  

        cv2.imshow( 'frame' + str(i), img )
        # try:
        #     cv2.imshow( 'frame' + str(i), frame )
        # except:
        #     continue

    # ESC  
    k = cv2.waitKey(20)
    if k == 27:
        # pygame.mixer.init()
        # pygame.mixer.music.load("./bgm/Countdown06-2.mp3")
        # pygame.mixer.music.play(1)
        # time.sleep(6)
        # pygame.mixer.music.stop()
        break
    elif k == ord('s'):
        executor.submit(func1)


capture.release()
cv2.destroyAllWindows()

 
