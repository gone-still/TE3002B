# File        :   classify.py (Pokenet's testing script)
# Version     :   1.0.0
# Description :   Script that calls the Pokenet CNN and tests it on example images.
# Date:       :   Feb 28, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

# Import the necessary packages:
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
from imutils import paths


# Reads image via OpenCV:
def readImage(imagePath):
    # Open image:
    print("readImage>> Reading: " + imagePath)
    inputImage = cv2.imread(imagePath)
    # showImage("Input Image", inputImage)

    if inputImage is None:
        print("readImage>> Could not load Input image.")

    return inputImage


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Set the resources paths:
mainPath = "D://CNN//pokenet//"
examplesPath = mainPath + "examples//"
modelPath = mainPath + "output//"

# Training image size:
imageSize = (75, 75)

# The class dictionary:
# bulbasaur		10000
# charmander	01000
# mewtwo		00100
# pikachu		00010
# squirtle		00001
classDictionary = {0: "bulbasaur", 1: "charmander", 2: "mewtwo", 3: "pika", 4: "squirtle"}

# Load model:
print("[Pokenet - Test] Loading network...")
model = load_model(modelPath + "pokenet.model")
lb = pickle.loads(open(modelPath + "labels.pickle", "rb").read())

# Get the test images paths:
imagePaths = sorted(list(paths.list_images(examplesPath)))

# Loop over the test images and classify each one:
for imagePath in imagePaths:

    # Load the image via OpenCV:
    image = readImage(imagePath)

    # Deep copy for displaying results:
    output = image.copy()

    # Pre-process the image for classification
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, imageSize)
    image = image.astype("float") / 255.0

    # Add the "batch" dimension:
    image = np.expand_dims(image, axis=0)

    # Classify the input image
    print("[Pokenet - Test] Classifying image...")

    # Send to CNN, get probabilities:
    predictions = model.predict(image)

    # Get max probability, thus, the max classification result:
    classIndex = predictions.argmax(axis=1)[0]
    # label = lb.classes_[classIndex] # Get categorical label via lb object
    label = classDictionary[classIndex] # Get categorical label via the class dictionary

    # Print the classification result:
    print("Class: " + label + " prob: " + str(predictions[0][classIndex]))

    # Build the label and draw the label on the image
    prob = "{:.2f}%".format(predictions[0][classIndex] * 100)
    label = label + " " + prob

    # New image dimensions for displaying results:
    (imageHeight, imageWidth) = output.shape[:2]
    scale = 60
    width = int(imageWidth * scale / 100)
    height = int(imageHeight * scale / 100)

    # Resize:
    output = cv2.resize(output, (width, height))

    # Draw Text:
    textColor = (155, 5, 170)
    cv2.putText(output, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)

    # Show the output image and its label & probability
    showImage("Output", output)
