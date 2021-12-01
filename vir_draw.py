import numpy as np
import cv2 as cv
import os
import handDeModule as htm
import torch
import torchvision.transforms as transforms
from digit_recog import neural_1

flag = True


brushThickness = 50
eraserThickness = 50

folder_path = "Header"

#loading the digit recogniser model
model_pth ="./trained_max_pool.pth"
#model_pth ="C:/Users/LENOVO/Desktop/Digitrecogniser/custom_digits/trained_1_81.pth"

model = neural_1()

model.load_state_dict(torch.load(model_pth))
model.eval()
transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])



myList = os.listdir(folder_path)
print(myList)
overlayList = []
for imPath in myList:
    image = cv.imread(f'{folder_path}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[1]
drawColor = (255,0,255)

cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.85)
xp,yp = 0,0
imgCanvas = np.ones((720,1280,3),np.uint8)
down_width = 28

down_height = 28
down_points = (down_width, down_height)


while True:
    #1. importing image
    success, img = cap.read()
    img = cv.flip(img,1)
    
    #2.find handlandmarks
    img = detector.findHands(img)
    lmList = detector.handPos(img,draw=False)
    
    if len(lmList)>0:
        #tip of index and middle finger
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
           
    #3.chacking finger up
    
        fingers = detector.fingersUp()
        
    #4.selection mode two fingers up
        if fingers[1] and fingers[2]:
            print("selection mode")
            xp,yp = 0,0
       
            if y1 < 125:
                if 0<x1<200:
                    header = overlayList[1]
                    drawColor = (200,50,200)
                if 200<x1<400:
                    header = overlayList[0]
                    drawColor = (0,0,0)
            cv.rectangle(img,(x1-15,y1-15),(x1+15 ,y1+25),drawColor,cv.FILLED)

        if fingers[1] and fingers[2]==False:
            cv.circle(img,(x1,y1),15,drawColor,cv.FILLED)
            print("writing mode")
            if xp==0 and yp == 0:
                xp,yp=x1,y1
            if drawColor == (0,0,0):
                cv.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            cv.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
            cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp = x1,y1
            
        #change the 3 channel to single channel
            
        imgGray = cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)
        
        if fingers[4]==1:
            # pred_img = imgGray[107:,107:1000]
            # pred_img = cv.resize(pred_img, (28,28), interpolation= cv.INTER_LINEAR)
            # _,pred_img = cv.threshold(pred_img,50,255,cv.THRESH_BINARY)
           
            pred_img = imgGray[107:,0:1280]
            pred_img = cv.resize(pred_img,(64,64),interpolation=cv.INTER_AREA)
            
            pred_img = transform(pred_img)
            pred_img = pred_img[None,:,:]
            out = model(pred_img)
            out = out.max(1).indices
            print(out)
            
    imgGray = cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)
    
    _, imgInv = cv.threshold(imgGray,50,255,cv.THRESH_BINARY_INV)   
    imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img,imgInv)
    img = cv.bitwise_or(img,imgCanvas)
    #5.drawing mode only index finger
    
    #overlaying the videoimage
    img[0:107,0:1280] = header
    # img = cv.addWeighted(img,0.5,imgInv,0.5,0)
    
    

    #cv.imshow("inv_img",pred_img)
    cv.imshow("image",img)
    #cv.imshow("Canvas",imgCanvas)
    
    cv.waitKey(1)
    #Low critical temperature (typically in the range of 0K to 10K)