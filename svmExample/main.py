# File        :   main.py (Classifier workflow example)
# Version     :   1.0.0
# Description :   Script that  shows a classic machine learning classifier
#                 workflow implementing a SVM for shape classification
# Date:       :   Feb 08, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT


# Imports:
import numpy as np
import cv2
import random

# import helper functions:
import imageUtils

# Compute shape attributes:
def computeAttributes(sampleList, attributtes):

    # Get list dimensions:
    numberOfSamples = len(sampleList)

    # Prepare the out array:
    outArray = np.zeros((numberOfSamples, attributtes + 1), dtype=np.float32)

    # Attribute computation:
    for i in range(len(sampleList)):
        # Get tuple from list:
        currentTuple = sampleList[i]
        # Get class (string):
        currentClass = currentTuple[0]
        # Get class (number):
        currentClassNumber = currentTuple[1]
        # Get current image:
        currentImage = currentTuple[2]

        # Set class:
        outArray[i][0] = currentClassNumber

        # To Gray:
        grayImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)
        # imageUtils.showImage("Gray", grayImage)

        # Threshold:
        _, binaryImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_OTSU)
        # imageUtils.showImage("Binary", binaryImage)

        # Get contours:
        contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # print("Contours: " + str(len(contours)))

        # Check out the contours, compute attributes:
        for _, c in enumerate(contours):

            # Get the shape's boundig box:
            x, y, w, h = cv2.boundingRect(c)

            # Aspect Ratio:
            aspectRatio = h / w

            # Circularity:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            circularity = pow(perimeter, 2) / (4 * 3.1416 * area)

            # Rotated Rect area:
            rect = cv2.minAreaRect(c)
            cx, cy = rect[0]
            w, h = rect[1]
            theta = rect[2]

            # Area Difference -> rotatedArea - area:
            rotatedRectArea = w * h
            areaDifference = rotatedRectArea - area

            print(currentClass + "A: " + str(aspectRatio) + " C: " + str(circularity) + " D: " + str(areaDifference))

            # Take those 3 attributes and shove them up into the array:
            outArray[i][1] = aspectRatio
            outArray[i][2] = circularity
            outArray[i][3] = areaDifference

    return outArray


# Set image path
path = "D://opencvImages//"

# Create the test/train lists:
trainList = []
testList = []

# Roll the PRNG:
random.seed(42)

# Create the class list and dictionary:
shapeClasses = ["circle", "square", "rectangle"]
classesDictionary = {0: "circle", 1: "square", 2: "rectangle"}

# Set the number of train/test samples:
trainSamples = 50
testSamples = 20

# Create the train data set:
for c in range(len(shapeClasses)):
    # Get class as string:
    currentClass = shapeClasses[c]
    # Get class as number 0-2:
    classNumber = list(classesDictionary.keys())[list(classesDictionary.values()).index(currentClass)]
    for s in range(trainSamples):
        # Create image
        outImage = imageUtils.createShape(currentClass, (200, 320))
        # Into the list:
        trainList.append((currentClass, classNumber, outImage))
        # imageUtils.showImage("Class: "+currentClass+", Sample: "+str(s), outImage)

# Create the test data set:
for c in range(testSamples):
    # Get random class as number:
    randomClass = random.randint(0, len(shapeClasses) - 1)
    # Get class as string:
    currentClass = classesDictionary[randomClass]

    # Create image
    outImage = imageUtils.createShape(currentClass, (200, 320))
    # Into the list:
    testList.append((currentClass, randomClass, outImage))
    print(currentClass)
    # imageUtils.showImage("Current Shape", outImage)

# Create the train/test matrices:
trainMatrix = computeAttributes(trainList, attributtes=3)
testMatrix = computeAttributes(testList, attributtes=3)

# Get train data and labels:
(trSamples, attributes) = trainMatrix.shape

# Get train labels:
trainLabels = trainMatrix[0:trSamples, 0:1].astype(np.int32)
# Get train data:
trainData = trainMatrix[0:trSamples, 1:attributes].astype(np.float32)

# Get test data and labels:
(tsSamples, attributes) = testMatrix.shape

# Get test labels:
testLabels = testMatrix[0:trSamples, 0:1].astype(np.int32)
# Get test data:
testData = testMatrix[0:trSamples, 1:attributes].astype(np.float32)

# Create the SVM:
SVM = cv2.ml.SVM_create()

# Set hyperparameters:
SVM.setKernel(cv2.ml.SVM_LINEAR)  # Sets the SVM kernel, this is a linear kernel
SVM.setType(cv2.ml.SVM_NU_SVC)    # Sets the SVM type, this is a "Smooth" Classifier
SVM.setNu(0.09)                   # Sets the "smoothness" of the decision boundary, values: [0.0 - 1.0]

SVM.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 25, 1.e-01))
SVM.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)

# Test:
# Begin Prediction:
svmResult = SVM.predict(testData)[1]

# Show accuracy:
mask = svmResult == testLabels
correct = np.count_nonzero(mask)
print("SVM Accuracy: " + str(correct * 100.0 / svmResult.size) + " %")

# Show each test sample and its classification result:
testImages = testLabels.shape[:1][0]

for s in range(testImages):

    # Get the current test image:
    currentImage = testList[s][2]
    # Get the SVM class for this test image:
    svmPrediction = svmResult[s][0]
    # Get class string based on class number:
    svmLabel = classesDictionary[svmPrediction]

    # Get real class number:
    realClass = testLabels[s][0]
    # Ger real string based on class number:
    realLabel = classesDictionary[realClass]

    # Draw labels on image:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(currentImage, "SVM: " + str(svmLabel), (3, 15), font, 0.5, (0, 255, 0), 1, cv2.LINE_8)
    cv2.putText(currentImage, "TRUTH: " + str(realLabel), (3, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_8)

    # Show the classified image:
    imageUtils.showImage("Test Image", currentImage)
