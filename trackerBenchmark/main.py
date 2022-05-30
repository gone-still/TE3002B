# File        :   main.py (Tracker Benchmark)
# Version     :   1.1.0
# Description :   Tests different tracking algorithms. Tracks a blue object.
# Date:       :   Feb 15, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import cv2
import os


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey()


# Writes an image as png file:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # print("Wrote Image: " + imagePath)


# Carries out morphological filtering:
def morhologicalChain(binaryImage):
    # Set morph operation iterations:
    opIterations = 1
    # Set Structuring Element size:
    structuringElementSize = (3, 3)
    # Set Structuring element shape:
    structuringElementShape = cv2.MORPH_RECT
    # Get the Structuring Element:
    structuringElement = cv2.getStructuringElement(structuringElementShape, structuringElementSize)

    # Perform Morpho Chain:
    morphedMask = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, structuringElement, None, None, opIterations,
                                   cv2.BORDER_REFLECT101)

    opIterations = 2
    morphedMask = cv2.morphologyEx(morphedMask, cv2.MORPH_ERODE, structuringElement, None, None, opIterations,
                                   cv2.BORDER_REFLECT101)
    # showImage("Morphed Mask 1", morphedMask)
    morphedMask = cv2.morphologyEx(morphedMask, cv2.MORPH_DILATE, structuringElement, None, None, opIterations,
                                   cv2.BORDER_REFLECT101)
    # showImage("Morphed Mask 2", morphedMask)

    return morphedMask


# Processes the frame for manual object detection:
def processFrame(inputFrame, minArea):
    # Return values:
    validBox = False
    bbox = []

    # Convert BGR to HSV:
    hsvFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2HSV)

    # Search for blue objects:
    lowerValues = np.array([104, 138, 112])
    upperValues = np.array([179, 255, 225])

    # Create HSV mask:
    hsvMask = cv2.inRange(hsvFrame, lowerValues, upperValues)
    # cv2.imshow("hsvMask", hsvMask)

    # Run morphological filtering:
    filteredMask = morhologicalChain(hsvMask)

    # Get contours:
    contours, _ = cv2.findContours(filteredMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):

        currentContour = contours[i]
        # Get the contour's bounding rectangle:
        boundRect = cv2.boundingRect(currentContour)

        # Get the dimensions of the bounding rect:
        rectX = boundRect[0]
        rectY = boundRect[1]
        rectWidth = boundRect[2]
        rectHeight = boundRect[3]

        rectArea = rectWidth * rectHeight
        print("i: " + str(i) + ", area: " + str(rectArea))

        # minArea = 300
        if rectArea > minArea:
            validBox = True
            # Store and set bounding rect:
            bbox = boundRect
            color = (0, 0, 255)
            cv2.rectangle(inputFrame, (int(rectX), int(rectY)),
                          (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)

            # showImage("Rectangles", inputFrame)

    return (validBox, bbox)


# Input path:
path = os.path.join("D:/", "opencvImages", "trackerBenchmark")
videoName = "testVideo01.mp4"

# Input flag.
# True = reads video, False = reads web cam
readVideo = True

# Tracker variables:
startTracking = False
bbox = []
tracker = None

# Specify the desired tracker here,
# select string from the tracker dictionary:
trackerString = "csrt"

# Tracker dictionary (available trackers):
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy_TrackerCSRT,
    "kcf": cv2.legacy_TrackerKCF,
    "boosting": cv2.legacy_TrackerBoosting,
    "mil": cv2.legacy_TrackerMIL,
    "tld": cv2.legacy_TrackerTLD,
    "medianflow": cv2.legacy_TrackerMedianFlow,
    "mosse": cv2.legacy_TrackerMOSSE
}

# Open Device:
if readVideo:
    videoPath = os.path.join(path, videoName)
    print("Reading video file: " + videoPath)
    cap = cv2.VideoCapture(videoPath)
else:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if device is opened:
if not (cap.isOpened()):
    print("Could not open video device")
else:
    print("Video Device opened")

    # Process frames:
    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if we have a valid frame:
        if not ret:
            print("Reached end of video file.")
            break

        # Resize frame:
        scale = 0.5
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dsize = (width, height)
        inputFrame = cv2.resize(frame, dsize, interpolation=cv2.INTER_AREA)
        # showImage("Resized", inputFrame)

        # Check tracker flag...
        if not (startTracking):
            # Run manual detection if the tracking is non-existant:
            minArea = scale * 1000
            (validBox, bbox) = processFrame(inputFrame, minArea)

            # Check if we have a good manual detection:
            if validBox:
                # Got good manual detection, initialize tracker:
                tracker = OPENCV_OBJECT_TRACKERS[trackerString].create()
                status = tracker.init(inputFrame, bbox)
                print("Starting Tracker...")
                startTracking = True
        else:
            # No need for manual detection, just update the tracker with a new
            # frame and get a nice new bounding box for the tracked object:
            status, bbox = tracker.update(inputFrame)
            # Status is a flag showing if the tracker lost the object:
            # print(status)
            if status:
                # Tracker is good, draw bounding rectangle:
                color = (0, 255, 0)
                cv2.rectangle(inputFrame, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), color, 2)

            else:
                # Tracker lost the object, re-run manual detection:
                print("Lost the object, running manual detection...")
                startTracking = False

        # Show Real-time tracking:
        cv2.imshow("Tracked", inputFrame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# When everything is done, release the capture device:
cap.release()
cv2.destroyAllWindows()
print("Video Device closed")
