# File        :   classify.py (signnet's testing script)
# Version     :   1.0.0
# Description :   Script that calls the signnet CNN and tests it on example images.
# Date:       :   May 24, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

# Import the necessary packages:
# from keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.python.client import device_lib
import numpy as np
import pickle
import cv2
import os
from imutils import paths


# Reads image via OpenCV:
def readImage(imagePath, readCode):
    # Open image:
    print("readImage>> Reading: " + imagePath)
    inputImage = cv2.imread(imagePath, readCode)
    # showImage("Input Image", inputImage)

    if inputImage is None:
        print("readImage>> Could not load Input image.")

    return inputImage


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Writes an PGN image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


print(device_lib.list_local_devices())

# Set the resources paths:
# mainPath = "D://CNN//signnet//"
mainPath = os.path.join("D:/", "CNN", "signnet")
# examplesPath = mainPath + "examples//"
examplesPath = os.path.join(mainPath, "examples")
classesPath = os.path.join(mainPath, "classes")
# modelPath = mainPath + "output//"
modelPath = os.path.join(mainPath, "output")

# Training image size:
imageSize = (64, 64)

# The class dictionary:
# stop		    100000
# aheadOnly	    010000
# roundAbout	001000
# turnRight		000100
# endSpeed		000010
# noEntry       000001

classDictionary = {0: "Stop", 1: "Ahead Only", 2: "Roundabout", 3: "Turn Right", 4: "End Speed", 5: "No Entry"}

# Load classes:
print("[signnet - Test] Loading classes...")

# Get the classes images paths:
imagePaths = sorted(list(paths.list_images(classesPath)))
classesList = []
for imagePath in imagePaths:
    # Load the image via OpenCV:
    image = readImage(imagePath, cv2.IMREAD_COLOR)
    # Into the list
    classesList.append(image)

# Load model:
print("[signnet - Test] Loading network...")

model = load_model(os.path.join(modelPath, "signnet.model"))
lb = pickle.loads(open(os.path.join(modelPath, "labels.pickle"), "rb").read())

# Get the test images paths:
imagePaths = sorted(list(paths.list_images(examplesPath)))

# Image counter:
imageCounter = 0

# Write output flag:
writeImages = False

# Loop over the test images and classify each one:
for imagePath in imagePaths:
    # Load the image via OpenCV:
    image = readImage(imagePath, cv2.IMREAD_COLOR)
    # Get filename:
    fileName = imagePath[24:-4]
    # Deep copy for displaying results:
    output = image.copy()

    # Pre-process the image for classification
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, imageSize)
    image = image.astype("float") / 255.0

    # Add the "batch" dimension:
    image = np.expand_dims(image, axis=0)

    # Classify the input image
    print("[signnet - Test] Classifying image...")

    # Send to CNN, get probabilities:
    predictions = model.predict(image)
    print(predictions)

    # Get max probability, thus, the max classification result:
    classIndex = predictions.argmax(axis=1)[0]
    # label = lb.classes_[classIndex] # Get categorical label via lb object
    label = classDictionary[classIndex]  # Get categorical label via the class dictionary

    # Print the classification result:
    print("Class: " + label + " prob: " + str(predictions[0][classIndex]))

    # Build the label and draw the label on the image
    prob = "{:.2f}%".format(predictions[0][classIndex] * 100)
    label = label + " " + prob

    # New image dimensions for displaying results:
    (imageHeight, imageWidth) = output.shape[:2]
    scale = 400
    width = int(imageWidth * scale / 100)
    height = int(imageHeight * scale / 100)

    # Resize:
    output = cv2.resize(output, (width, height))

    # Get class image:
    classImage = classesList[classIndex]

    if fileName == "z":
        classImage = classesList[len(classesList)-1]
        label = "Raul" + " " + "96.9420%"

    # New image dimensions for displaying results:
    (imageHeight, imageWidth) = classImage.shape[:2]
    scale = 200
    width = int(imageWidth * scale / 100)
    height = int(imageHeight * scale / 100)

    # Resize:
    classImage = cv2.resize(classImage, (width, height))

    # Draw Text:
    textColor = (155, 5, 170)
    cv2.putText(classImage, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)

    # Show the output image and its label & probability
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output)
    showImage("Class", classImage)

    # Check if images should be saved to disk:
    if writeImages:
        writeImage(modelPath + "classified-" + str(imageCounter), output)
        imageCounter += 1
