import cv2 as cv
import mediapipe as mp
import time

class handDetector:
    def __init__(self,mode = False,maxHands = 2,detectionCon = 0.5,trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]
        
    def findHands(self,img,draw = True):
        img2 = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.resutls = self.hands.process(img2)
        if self.resutls.multi_hand_landmarks:
            for handslms in self.resutls.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handslms,self.mpHands.HAND_CONNECTIONS)
        return img
    
    def handPos(self, img,highlight=None, handsNo = 0 , draw = True):
        self.lmList = []
        if self.resutls.multi_hand_landmarks:
            hand=self.resutls.multi_hand_landmarks[handsNo]
            for id,lm in enumerate(hand.landmark):
                        h,w,c = img.shape
                        cx,cy = int(lm.x*w),int(lm.y*h)
                        self.lmList.append([id,cx,cy])
                        if draw:
                            if id == highlight:
                                cv.circle(img,(cx,cy),25,(0,0,255),3)
        return self.lmList
    
    def fingersUp(self):
        fingers = []
        
        #thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        #4 fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers





def main():
    ptime = 0
    ctime = 0
    obj = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success,img = obj.read()
        img = detector.findHands(img)   
        lmList = detector.handPos(img)
        
        if len(lmList)!=0:
            print(lmList[4])
        
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        
        cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
        
        cv.imshow("image",img)
        cv.waitKey(1)
    
    
    
if __name__ == '__main__':
    main()