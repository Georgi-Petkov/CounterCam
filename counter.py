import datetime
import math
import cv2
import numpy as np
import imutils

#global variables
width = 0
height = 0
entranceCounter = 0
exitCounter = 0
minContourArea = 3000  #Adjust this value according to your usage
binarizationThreshold = 70  #Adjust this value according to your usage
offsetRefLines = 150  #Adjust this value according to your usage

#Check if an object in entering in monitored zone
def CheckEntranceLineCrossing(y, coordYEntranceLine, coordYExitLine):
    absDistance = abs(y - coordYEntranceLine)    

    if ((absDistance <= 2) and (y < coordYExitLine)):
        return 1
    else:
        return 0

#Check if an object in exiting from monitored zone
def CheckExitLineCrossing(y, coordYEntranceLine, coordYExitLine):
    absDistance = abs(y - coordYExitLine)    

    if ((absDistance <= 2) and (y > coordYEntranceLine)):
        return 1
    else:
        return 0

camera = cv2.VideoCapture(0)

#force 640x480 webcam resolution
camera.set(3,640)
camera.set(4,480)

referenceFrame = None

#The webcam maybe get some time / captured frames to adapt to ambience lighting. For this reason, some frames are grabbed and discarded.
for i in range(0,20):
    (grabbed, frame) = camera.read()

while True:    
    (grabbed, frame) = camera.read()
    height = np.size(frame,0)
    width = np.size(frame,1)

    #if cannot grab a frame, this program ends here.
    if not grabbed:
        break

    #gray-scale convertion and Gaussian blur filter applying
    GrayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    GrayFrame = cv2.GaussianBlur(GrayFrame, (21, 21), 0)
    
    if referenceFrame is None:
        referenceFrame = GrayFrame
        continue

    #Background subtraction and image binarization
    frameDelta = cv2.absdiff(referenceFrame, GrayFrame)
    frameThresh = cv2.threshold(frameDelta, binarizationThreshold, 255, cv2.THRESH_BINARY)[1]
    
    #Dilate image and find all the contours
    frameThresh = cv2.dilate(frameThresh, None, iterations=2)
    #_, cnts, _ = cv2.findContours(frameThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cv2.findContours(frameThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    qtyOfContours = 0

    #plot reference lines (entrance and exit lines) 
    coordYEntranceLine = round((height / 2)-offsetRefLines) # rounded to get int value
    coordYExitLine = round((height / 2)+offsetRefLines)
    cv2.line(frame, (0,coordYEntranceLine), (width,coordYEntranceLine), (255, 0, 0), 2)
    cv2.line(frame, (0,coordYExitLine), (width,coordYExitLine), (0, 0, 255), 2)


    #check all found contours
    for c in cnts:
        #if a contour has small area, it'll be ignored
        if cv2.contourArea(c) < minContourArea:
            continue

        qtyOfContours = qtyOfContours+1    

        #draw a rectangle "around" the object
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #find object's centroid
        coordXCentroid = (x+x+w)/2
        coordYCentroid = (y+y+h)/2
        ObjectCentroid = (round(coordXCentroid), round(coordYCentroid)) # rounded to get int value
        cv2.circle(frame, ObjectCentroid, 1, (0, 0, 0), 5)
        
        if (CheckEntranceLineCrossing(coordYCentroid,coordYEntranceLine,coordYExitLine)):
            entranceCounter += 1

        if (CheckExitLineCrossing(coordYCentroid,coordYEntranceLine,coordYExitLine)):  
            exitCounter += 1

    print ("Total contours found: " + str(qtyOfContours))

    #Write entrance and exit counter values on frame and shows it
    cv2.putText(frame, "Entrances: {}".format(str(entranceCounter)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 1), 2)
    cv2.putText(frame, "Exits: {}".format(str(exitCounter)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Original frame", frame)
    cv2.waitKey(1);


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()