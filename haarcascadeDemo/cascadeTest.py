# File        :   cascadeTest.py (Haar Cascade Test)
# Version     :   1.1.1
# Description :   Script that implements a Haar Cascade Classifier for
#             :   traffic signal detection
# Date:       :   May 29, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import cv2
import os


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


# Main Resources Path:
# Change paths accordingly, goddamnit!
filePath = os.path.join("D:/", "trackerTest")

# Load cascade:
modelsPath = os.path.join(filePath, "models")
cascadePath = os.path.join(modelsPath, "cascades", "signalCascade-01.xml")
signCascade = cv2.CascadeClassifier(cascadePath)

# Read image:
inputImage = cv2.imread(os.path.join(filePath, "inputFrame.png"))

# BGR to Gray:
# (Haar Cascade Receives a grayscale image)
grayImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Configure Haar Cascade
# See https://docs.opencv.org/4.5.5/db/d28/tutorial_cascade_classifier.html
boundingBoxes = signCascade.detectMultiScale(grayImage,
                                             scaleFactor=1.015,
                                             minNeighbors=15,
                                             minSize=(3, 3))
# Get total of detected bounding boxes:
totalBoxes = len(boundingBoxes)
print("Objects detected via Cascade: " + str(totalBoxes))

# Loop through detected boxes:
for (x, y, w, h) in boundingBoxes:
    # Set color:
    color = (0, 255, 0)
    # Draw bounding box:
    cv2.rectangle(inputImage, (x, y), (x + w, y + h), color, 2)
    # Show image:
    showImage("Haar Cascade Boxes", inputImage)
