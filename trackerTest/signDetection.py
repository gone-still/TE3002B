# File        :   signDetection.py (Traffic signal detection, classification and tracking example)
# Version     :   0.10.2
# Description :   Script that tests classification + tracking of
#             :   traffic signal
# Date:       :   June 04, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import cv2
import numpy as np
from fastKLT import FastKLT
from tensorflow.keras.models import load_model


# Shows an image
def showImage(imageName, inputImage, delay=0):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(delay)


# Writes a png image to disk:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


# Clamps an integer to a valid range:
def clamp(val, minval, maxval):
    if val < minval: return minval
    if val > maxval: return maxval
    return val


# Obtains a blob bounding rect via grab-cut
def getGrabCutMask(inputRect, inputImage):
    # Unpack the rect tuple:
    (sx, sy, sw, sh) = inputRect

    # Default out values:
    goodBlob = False
    blobRect = ()
    # goodMask = np.zeros((sh, sw, 1), dtype="uint8")

    # Define object area for grab-cut (the "window"),
    # Window centroid:
    cxWindow = int(0.5 * sw)
    cyWindow = int(0.5 * sh)

    # Loop thru all window scales:
    for s in range(1, 4):
        # Get current scale:
        s = 2 * s
        print("scale: " + str(s))

        # Define window top left corner:
        currentWindowScale = 1 / s
        xWindow = int(currentWindowScale * cxWindow)
        yWindow = int(currentWindowScale * cyWindow)

        # Define window width and height:
        wWindow = int(2 * (cxWindow - xWindow))
        hWindow = int(2 * (cyWindow - yWindow))

        # Define the tuple:
        grabCutRect = (xWindow, yWindow, wWindow, hWindow)
        print(grabCutRect)

        # Show the window:
        grabCutArea = inputImage.copy()

        # Show the grab cut area:
        color = (0, 0, 255)
        cv2.rectangle(grabCutArea, (xWindow, yWindow), (xWindow + wWindow, yWindow + hWindow),
                      color, 2)

        # Show centroid:
        color = (255, 0, 0)
        cv2.line(grabCutArea, (cxWindow, cyWindow), (cxWindow, cyWindow), color, 2)

        showImage("grabCutArea: " + str(s), grabCutArea)
        # writeImage(outPath + "grabCutArea", grabCutArea)

        # Tune the detection using grab n cut:
        # The mask is a uint8 type, same dimensions as
        # original input:
        mask = np.zeros(inputImage.shape[:2], np.uint8)

        # Grab n Cut needs two empty matrices of
        # Float type (64 bits) and size 1 (rows) x 65 (columns):
        bgModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)

        # Run Grab n Cut on INIT_WITH_RECT mode:
        grabCutIterations = 2
        mask, bgModel, fgModel = cv2.grabCut(inputImage, mask, grabCutRect, bgModel, fgModel,
                                             grabCutIterations, mode=cv2.GC_INIT_WITH_RECT)

        # Set all definite background (0) and probable background pixels (2)
        # to 0 while definite foreground and probable foreground pixels are
        # set to 1
        outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)

        # Scale the mask from the range [0, 1] to [0, 255]
        outputMask = (outputMask * 255).astype("uint8")

        showImage("GrabCut Mask", outputMask)
        # writeImage(outPath + "grabCutMask", outputMask)

        # Get blob area:
        currentBlobArea = cv2.countNonZero(outputMask)
        print("currentBlobArea: " + str(currentBlobArea))

        # Check if we have a good blob,
        # Check area:
        if currentBlobArea > minBlobArea:

            # Get blob rect:
            (bx, by, bw, bh) = cv2.boundingRect(outputMask)
            blobAspectRatio = bh / bw
            print("blobAspectRatio: " + str(blobAspectRatio))

            # Get aspect ratio difference:
            aspectRatioDifference = abs(1.0 - blobAspectRatio)
            epsilon = 0.3

            # Check aspect ratio:
            if aspectRatioDifference <= epsilon:
                print("Got good blob. Scale: " + str(s))
                goodBlob = True
                # # Give some slack:
                # bx = clamp(bx - trackerBorders[0], 0, bw)
                # by = clamp(by - trackerBorders[1], 0, bh)
                # bw = clamp(bw + 2 * trackerBorders[2], 0, bw)
                # bh = clamp(bh + 2 * trackerBorders[3], 0, bh)
                blobRect = (bx, by, bw, bh)
                # goodMask = outputMask

    # Set out values:
    outTuple = (goodBlob, blobRect)

    return outTuple


# Script variables:

# Set the file paths and names:
filePath = "D://trackerTest//"
outPath = filePath + "out//"
modelsPath = filePath + "models//"

# CNN size:
imageSize = (64, 64)

# Class dictionary:
classDictionary = {0: "Stop", 1: "Ahead Only", 2: "Roundabout", 3: "Turn Right", 4: "End Speed", 5: "No Entry"}

# Speed of the video:
videoSpeed = 1
frameWidth = 1280
frameHeight = 720

# Frame aspect ratio:
aspectRatio = frameWidth / frameHeight

# Frame Counter:
frameCounter = 0

# Min CNN probability:
minClassProbability = 0.6

# Cascade resize parameters:
cascadeScale = 50

# Resize the frame for cascade detection:
resizedWidth = int(frameWidth * cascadeScale / 100)
resizedHeight = int(frameHeight * cascadeScale / 100)

# Cascade ROI
# Crop the roi for cascade detection, top left, width, height:
roiScale = cascadeScale / 100
roiX = int(0)
roiY = int(60)
roiHeight = int(255)
roiWidth = int(frameWidth * roiScale)
cascadeRoi = (int(roiX), int(roiY * roiScale), int(roiWidth), int(roiHeight))

# Left/Right starting horizontal coordinates:
sideWidthFactor = 0.4
leftSide = int(sideWidthFactor * roiWidth)
rightSide = int(roiWidth - sideWidthFactor * roiWidth)

# Detection mask:
detectionMask = np.zeros((resizedHeight, resizedWidth, 1), np.uint8)
detectionMask = 255 - detectionMask

# Set the detection mask coordinates:
maskX = leftSide
maskY = roiY
maskWidth = rightSide - leftSide
maskHeight = roiHeight

# draw detection mask rect:
cv2.rectangle(detectionMask, (maskX, maskY), (maskX + maskWidth, maskY + maskHeight), 0, -1)
showImage("detectionMask", detectionMask)

# Crop Mask to detection dimensions:
detectionMask = detectionMask[cascadeRoi[1]:cascadeRoi[1] + cascadeRoi[3],
                cascadeRoi[0]:cascadeRoi[0] + cascadeRoi[2]]
showImage("detectionMask [Cropped]", detectionMask)

# Tracker  Parameters:
maxFeatures = 100
fastThreshold = 5
nRows = 3
nCols = 3
kltWindowSize = 10
shrinkRatio = 0.05
ransacThreshold = 0.9
trackerId = 1

# Tracking margin (x,y,w,h):
trackerBorders = (2, 2, 2, 2)

# Running cascade at first frame:
runCascade = True

# Set tracker parameters:
parametersTuple = [maxFeatures, (nRows, nCols), fastThreshold, shrinkRatio, (kltWindowSize, kltWindowSize),
                   ransacThreshold, trackerId]

# Create the tracker with parameters:
tracker = FastKLT(parametersTuple)

# Enable debug information:
tracker.setVerbose(False)

# Show tracker's grid keypoints:
tracker.showGrid(False)

# Load the CNN model:
model = load_model(modelsPath + "signnet.model")
classString = ""

# Set the video device:
videoDevice = cv2.VideoCapture(filePath + "trafficSign05.mp4")

trackerCounter = 0

# Load cascade:
signCascade = cv2.CascadeClassifier(modelsPath + "cascades//" "signalCascade-04.xml")

# Threshold parameters:
minCascadeArea = 900
minCascadeAspectRatio = 0.9
minBlobArea = 10

# Check if device is opened:
while videoDevice.isOpened():
    # Get video device frame:
    success, frame = videoDevice.read()

    # We have a nice frame to process:
    if success:

        # Extract frame size:
        (frameHeight, frameWidth) = frame.shape[:2]

        # Resize image
        detectionRoi = cv2.resize(frame, (resizedWidth, resizedHeight), interpolation=cv2.INTER_LINEAR)
        # writeImage(filePath+"inputFrame", detectionRoi)

        # Resized deep copy:
        roiCopy = detectionRoi.copy()

        # Draw ROI area:
        # Roi rect:
        cv2.rectangle(roiCopy, (roiX, roiY), (roiX + roiWidth, roiY + roiHeight), (0, 0, 255), 1)

        # Left and right:
        cv2.line(roiCopy, (leftSide, 0), (leftSide, roiY + resizedHeight), (255, 0, 0), 1)
        cv2.line(roiCopy, (rightSide, 0), (rightSide, roiY + resizedHeight), (255, 0, 0), 1)
        showImage("roiCopy", roiCopy)

        # Crop to detection dimensions:
        detectionRoi = detectionRoi[cascadeRoi[1]:cascadeRoi[1] + cascadeRoi[3],
                       cascadeRoi[0]:cascadeRoi[0] + cascadeRoi[2]]

        # Grayscale Conversion:
        detectionRoiColor = detectionRoi.copy()
        detectionRoi = cv2.cvtColor(detectionRoi, cv2.COLOR_BGR2GRAY)
        showImage("detectionRoi", detectionRoi)

        # Let's see if we must run cascade detection:
        if runCascade:

            # Run Haar Cascade
            # Tune the parameters to tune detection;s quality:
            boundingBoxes = signCascade.detectMultiScale(detectionRoi, scaleFactor=1.015, minNeighbors=4,
                                                         minSize=(3, 3))

            totalBoxes = len(boundingBoxes)
            print("Objects detected via Cascade: " + str(totalBoxes))

            # We need at least one detection:
            if totalBoxes > 0:

                # Got detection,
                # Convert gray ROI to BGR:
                detectionRoi = cv2.cvtColor(detectionRoi, cv2.COLOR_GRAY2BGR)

                # Loop through all mah bounding boxes:
                for (x, y, w, h) in boundingBoxes:

                    # Compute box area:
                    cascadeArea = w * h
                    print("Cascade Area: " + str(cascadeArea))

                    # Check minimum area:
                    if cascadeArea >= minCascadeArea:
                        print("Got box with good area.")

                        # Compute box centroid:
                        cx = int(x + 0.5 * w)
                        cy = int(y + 0.5 * h)

                        # Default color:
                        color = (0, 0, 0)

                        # Get detection mask "zone valid" pixel:
                        validPixel = int(detectionMask[cy, cx])

                        # Check if we have a valid pixel inside the
                        # "processing" zone:
                        if validPixel == 255:
                            # green is right:
                            color = (0, 255, 0)
                            print("Got valid Haar Cascade Box")
                        else:
                            # red is not:
                            color = (0, 0, 255)
                            print("Got invalid Haar Cascade Box")

                        # Draw the bounding box:
                        cv2.rectangle(detectionRoi, (x, y), (x + w, y + h), color, 2)
                        showImage("Haar Boxes", detectionRoi)

                        # So far, so good. Continue processing:
                        if validPixel == 255:

                            # Crop via cascade:
                            targetCrop = detectionRoiColor[y:y + h, x:x + w]
                            showImage("targetCrop [Cascade]", targetCrop)

                            # Define the "search window" for
                            # grab-cut:
                            maskRect = (x, y, w, h)

                            # Get refined rectangle via grab-cut:
                            (goodBlob, boundRect) = getGrabCutMask(maskRect, targetCrop)

                            # Check out if grab-cut got a valid blob:
                            if goodBlob:

                                print("Grab-cut found valid blob.")

                                # Got good blob, compute its bounding rectangle:
                                # boundRect = cv2.boundingRect(outputMask)

                                # Set new rect dimensions:
                                xGrabCut = boundRect[0]
                                yGrabCut = boundRect[1]
                                wGrabCut = boundRect[2]
                                hGrabCut = boundRect[3]

                                # Refine crop area:
                                targetCrop = targetCrop[yGrabCut:yGrabCut + hGrabCut, xGrabCut:xGrabCut + wGrabCut]

                                showImage("targetCrop [Refined]", targetCrop)
                                # writeImage(outPath + "targetCropRefined", targetCrop)

                                print("Sending crop to CNN...")
                                showImage("targetCrop [Pre-process]", targetCrop)

                                # Resize to CNN dimensions:
                                targetCrop = cv2.cvtColor(targetCrop, cv2.COLOR_BGR2RGB)
                                targetCrop = cv2.resize(targetCrop, imageSize)
                                showImage("targetCrop [Post-process]", targetCrop)

                                # Scale between 0.0 and 1.0
                                targetCrop = targetCrop.astype("float") / 255.0

                                # Add the "batch" dimension:
                                targetCrop = np.expand_dims(targetCrop, axis=0)

                                print("[signnet - Test] Classifying image...")

                                # Get the goddamn predictions:
                                predictions = model.predict(targetCrop)
                                print(predictions)

                                # Get max probability and its class:
                                classIndex = predictions.argmax(axis=1)[0]
                                classLabel = classDictionary[classIndex]
                                classProbability = predictions[0][classIndex]
                                print("ClassIndex:", classIndex, " classProbability:", classProbability, " classLabel:",
                                      classLabel)

                                # Yeah, discard bullshit classifications and process
                                # only if we have a good prediction:
                                if classProbability >= minClassProbability:
                                    classString = str(classIndex) + " " + classLabel + " (" + str(
                                        int(100 * classProbability)) + "%)"
                                    print("Sending frame to tracker...")

                                    # Goes to the tracker:
                                    # Add the initial cropped amount and add some margins:
                                    print((x, y, w, h))
                                    (cropHeight, cropWidth) = detectionRoi.shape[:2]
                                    xTrack = clamp(x + (xGrabCut - trackerBorders[0]), 0, cropWidth)
                                    yTrack = clamp(y + (yGrabCut - trackerBorders[1]), 0, cropHeight)
                                    wTrack = clamp(wGrabCut + 2 * trackerBorders[0], 0, cropWidth)
                                    hTrack = clamp(hGrabCut + 2 * trackerBorders[1], 0, cropHeight)

                                    # My bounding rectangle, lemme show it to you:
                                    print((xTrack, yTrack, wTrack, hTrack))

                                    # Draw the trackin area:
                                    color = (255, 0, 255)
                                    trackerRectInput = detectionRoiColor.copy()
                                    detectionRoi = cv2.rectangle(trackerRectInput, (int(xTrack), int(yTrack)),
                                                                 (int(xTrack + wTrack), int(yTrack + hTrack)), color, 2)
                                    showImage("trackerRectInput [Tracker Rect]", trackerRectInput)

                                    # showImage("detectionRoi [Tracker Input]", detectionRoi)
                                    # writeImage(outPath + "trackerInput", detectionRoi)

                                    # Fucking initialize the tracker:
                                    tracker.initTracker(detectionRoi, (xTrack, yTrack, wTrack, hTrack))
                                    # Cascade is no longer needed:
                                    runCascade = False
                                else:
                                    # I got bullshit, courtesy of the CNN:
                                    print("Min Class Probability not met. Running CNN on next frame...")

        else:

            print("Updating Tracker...")

            # Update the tracker:
            detectionRoi = cv2.cvtColor(detectionRoi, cv2.COLOR_GRAY2BGR)
            status, trackedObj = tracker.updateTracker(detectionRoi)

            print(status)

            # If the tracker is good, let's continue
            # processing:
            if status:
                # Draw rectangle:
                (startX, startY, endX, endY) = trackedObj
                color = (0, 255, 0)
                detectionRoi = cv2.rectangle(detectionRoi, (int(startX), int(startY)),
                                             (int(startX + endX), int(startY + endY)), color, 2)
                # Class text:
                org = (int(startX), int(startY + endY))
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (255, 0, 0)
                cv2.putText(detectionRoi, classString, org, font, 0.4, color, 1, cv2.LINE_AA)

            else:
                # Tracker failed (probably lost or not enough keypoints),
                # Run manual detection + classification on next frames:
                runCascade = True

        # Show the final output:
        showImage("resizedImage [Objects]", detectionRoi, 0)

        # writeImage(outPath + "detectionRoi-" + str(trackerCounter), detectionRoi)
        trackerCounter += 1

        # Increase frame counter:
        frameCounter += 1

        # Show the raw, input frame:
        textX = 10
        textY = 30
        org = (textX, textY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)
        frameString = "Frame: " + str(frameCounter)
        cv2.putText(frame, frameString, org, font, 1, color, 1, cv2.LINE_AA)
        showImage("Input Frame", frame, videoSpeed)
        # writeImage(outPath + "inputFrame", frame)

        # Break on "q"
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:

        print("Could not extract frame.")
        break

# Release the capture device:
videoDevice.release()
cv2.destroyAllWindows()
print("Video Device closed")
