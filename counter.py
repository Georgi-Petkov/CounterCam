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
binarizationThreshold = 55  #Adjust this value according to your usage
offsetRefLines = 260  # From center of frame
pixelTolerance = 3 # Tolerance zone width(/2) in pixels
coordXEntranceLine = 0
coordXExitLine = 0

#Check if an object in entering in monitored zone
def CheckEntranceLineCrossing(x, coordXEntranceLine, coordXExitLine):
    absDistance = abs(x - coordXEntranceLine)    

    if ((absDistance <= pixelTolerance) and (x < coordXExitLine)):
        return 1
    else:
        return 0

#Check if an object in exiting from monitored zone
def CheckExitLineCrossing(x, coordXEntranceLine, coordXExitLine):
    absDistance = abs(x - coordXExitLine)    

    if ((absDistance <= pixelTolerance) and (x > coordXEntranceLine)):
        return 1
    else:
        return 0

camera = cv2.VideoCapture(0)

#force 640x480 webcam resolution
camera.set(3,640)
camera.set(4,480)

referenceFrame = None

#The webcam maybe get some time / captured frames to adapt to ambience lighting. 
#For this reason, some frames are grabbed and discarded.
for i in range(0,20):
    (grabbed, frame) = camera.read()
    height = np.size(frame,0)
    width = np.size(frame,1)
    
    #refline positions (rounded to get int value)
    coordXEntranceLine = round((width / 2) - offsetRefLines) 
    coordXExitLine = round((width / 2) + offsetRefLines)

while True:    
    (grabbed, frame) = camera.read()

    #if cannot grab a frame, this program ends here.
    if not grabbed:
        break

    #gray-scale convertion and Gaussian blur filter applying
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayFrame = cv2.GaussianBlur(grayFrame, (21, 21), 0)
    
    if referenceFrame is None:
        referenceFrame = grayFrame
        continue

    #Background subtraction and image binarization
    frameDelta = cv2.absdiff(referenceFrame, grayFrame)
    frameThresh = cv2.threshold(frameDelta, binarizationThreshold, 255, cv2.THRESH_BINARY)[1]
    
    #Dilate image and find all the contours
    frameThresh = cv2.dilate(frameThresh, None, iterations=2)
    cnts = cv2.findContours(frameThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    qtyOfContours = 0

    #plot reference lines (entrance and exit lines) 
    cv2.line(frameThresh, (coordXEntranceLine,0), (coordXEntranceLine,height), (255, 0, 0), 2)
    cv2.line(frameThresh, (coordXExitLine,0), (coordXExitLine,height), (255, 0, 0), 2)

    #check all found contours
    for c in cnts:
        #if a contour has small area, it'll be ignored
        if cv2.contourArea(c) < minContourArea:
            continue

        qtyOfContours += 1 

        #draw a rectangle "around" the object
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frameThresh, (x, y), (x + w, y + h), (255, 255, 255), 2)

        #find object's centroid (rounded to get int value)
        coordXCentroid = round(x+w/2)
        coordYCentroid = round(y+h/2)
        cv2.circle(frameThresh, (coordXCentroid, coordYCentroid), 1, (255, 255, 255), 5)
        
        if (CheckEntranceLineCrossing(coordXCentroid,coordXEntranceLine,coordXExitLine)):
            entranceCounter += 1

        if (CheckExitLineCrossing(coordXCentroid,coordXEntranceLine,coordXExitLine)):  
            exitCounter += 1

    print ("Total contours found: " + str(qtyOfContours))

    #Write entrance and exit counter values on frame and show it
    cv2.putText(frameThresh, "Entrances: {}".format(str(entranceCounter)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 1), 2)
    cv2.putText(frameThresh, "Exits: {}".format(str(exitCounter)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 1), 2)
    cv2.imshow("Background substracted image", frameThresh)
    #cv2.imshow("Original frame", frame)
    cv2.waitKey(1);


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
