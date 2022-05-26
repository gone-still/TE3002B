# File        :   train.py (signnet's training script)
# Version     :   1.0.0
# Description :   Script that calls the Signnet CNN and trains it.
# Date:       :   May 26, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

# Import the signnet:
from signnet import signnet

# Import keras modules:
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

# Label binarizer:
from sklearn.preprocessing import LabelBinarizer

# The dataset split package:
from sklearn.model_selection import train_test_split

# Import matplot for plotting:
import matplotlib
import matplotlib.pyplot as plt

# Set matplotlib's renderer:
matplotlib.use("Agg")

# Path management:
from imutils import paths

# Other packages:
import numpy as np
import random
import pickle
import cv2
import os
from datetime import datetime
import os


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Training early stop callback.
# The callback is called when a epoch ends:
class stopTrainCallback(Callback):
    def __init__(self, thresholdTuple, counter):
        super(stopTrainCallback, self).__init__()
        # Set the threshold value for stopping
        (threshold, thresholdLimit) = thresholdTuple
        self.threshold = threshold
        self.thresholdLimit = thresholdLimit
        self.thresholdCount = 1

        # Set the epoch counter:
        self.epochCounter = counter

    def on_epoch_end(self, epoch, logs=None):
        # Increase the epoch counter:
        self.epochCounter += 1
        # The value I'm monitoring is the validation loss:
        accuracy = logs["val_loss"]
        # print(" Monitor: " + str(accuracy))
        # If threshold (min value) reached, end training:
        if accuracy <= self.threshold:
            if self.thresholdCount < self.thresholdLimit:
                self.thresholdCount += 1
                print(" stopTrainCallback>> Count: " + str(self.thresholdCount))
            else:
                print(" stopTrainCallback>> Got count limit.")
                global epochs
                epochs = self.epochCounter
                self.model.stop_training = True


# Set the resources paths:

# mainPath = "D://CNN//signnet//"  # Main directory
mainPath = os.path.join("D:/", "CNN", "signnet")
# dataSetPath = mainPath + "dataset"  # Dataset path
dataSetPath = os.path.join(mainPath, "dataset")
# outputPath = mainPath + "output"  # Output directory
outputPath = os.path.join(mainPath, "output")

# Training hyper parameters:
epochs = 95
alpha = 0.8899 # 0.5  # Learning rate factor
learningRate = alpha * 0.0001  # 2e-3, (0.001, 32)
batchSize = 32  # Could be: 16, 32, 64 In general, the smaller the better
imageDimensions = (64, 64, 3)  # The image is resized to these dimensions
stopThreshold = (0.15, 3)  # Thresh and count

# Callback epoch counter:
epochCounter = 0

# initialize the data and labels
data = []
labels = []

# Image counter:
imageCounter = 0

# Load each image path of the dataset:
print("[signnet - Train] Loading images...")
imagePaths = sorted(list(paths.list_images(dataSetPath)))

# Randomly shuffle the paths:
print("[signnet - Train] Shuffling images...")

# First time...:
random.seed(datetime.now())
random.shuffle(imagePaths)

# ...and again:
random.seed(datetime.now())
random.shuffle(imagePaths)

# Loop over the input images and load em:
for imagePath in imagePaths:
    # print("[signnet] Loading: "+imagePath)

    # Read the image via OpenCV:
    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    # Reorder the channels from BGR (OpenCV's default) to RGB:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Show the image:
    # showImage("Input Image", image)

    # Resize the image to CNN dimensions:
    image = cv2.resize(image, (imageDimensions[1], imageDimensions[0]))

    # The image goes into the list:
    data.append(image)
    imageCounter += 1

    # Extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# Convert lists to numpy arrays and scale the
# raw pixel intensities to the range [0, 1]:
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Label encoding for train_test_split:
# stop		    100000
# aheadOnly	    010000
# roundAbout	001000
# turnRight		000100
# endSpeed		000010
# noEntry       000001

# One-hot encode the labels:
lb = LabelBinarizer()
labelsEncoded = lb.fit_transform(labels)

# Partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labelsEncoded, test_size=0.20, random_state=666)

print("[signnet - Train] Images Loaded: " + str(imageCounter))
print("[signnet - Train] Train Samples: " + str(trainX.shape[0]))
print("[signnet - Train] Test Samples: " + str(testX.shape[0]))

# Check out the sample distribution. We can indicate the model
# which class is under represented, so it can handle it accordingly.
# This vector holds all the samples per class:
classTotal = trainY.sum(axis=0)
print("[signnet - Train] Sample count per class: ", classTotal)

# Class weight calculation:
classWeight = classTotal.max() / classTotal

# Create a dictionary that stores the weights per class:
classWeight = {i: classWeight[i] for i in range(len(classWeight))}

# Dataset Augmentation:
# Construct the image generator for data augmentation:

aug = ImageDataGenerator(
    rotation_range=5,
    zoom_range=0.3,
    width_shift_range=0.20,
    height_shift_range=0.20,
    # brightness_range=[0.9, 1.1],
    shear_range=0.35,
    # horizontal_flip=True,
    fill_mode="nearest")

# Checking out the Augmented Samples:
# The .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
# i = 0
# for batch in aug.flow(trainX, batch_size=1,
#                       save_to_dir="preview", save_prefix="signs", save_format='jpeg'):
#     i += 1
#     if i > 11:
#         break  # otherwise the generator would loop indefinitely
#
# # Initialize the model
print("[signnet - Train] Setting up the CNN...")

# Set up the CNN:
model = signnet.build(width=imageDimensions[1], height=imageDimensions[0], depth=imageDimensions[2],
                      classes=len(classTotal))

# Set up the optimizer
cnnOptimizer = Adam(learning_rate=learningRate,
                    beta_1=0.95,
                    beta_2=0.999,
                    # epsilon=1e-08
                    # decay=learningRate / (epochs * 0.5)
                    )

# Compile the model:
model.compile(loss="categorical_crossentropy", optimizer=cnnOptimizer, metrics=["accuracy"])

# Train the network
print("[signnet - Train] Training CNN...")

# Set early stop callback on validation loss with
# a threshold equal to or lower than 0.19:
callback = stopTrainCallback(thresholdTuple=stopThreshold, counter=epochCounter)

# Actually train the network:
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batchSize),
    validation_data=(testX, testY),
    # steps_per_epoch=trainX.shape[0] // batchSize,
    epochs=epochs,
    class_weight=classWeight,
    verbose=1 #,
    # callbacks=[callback]
    )

# Save the model to disk:
print("[signnet - Train] Saving model to disk...")
model.save(outputPath + "//signnet.model")

# Save the label binarizer to disk:
print("[signnet - Train] Serializing label binarizer...")
f = open(outputPath + "//labels.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()

# Get the historical data:
N = np.arange(0, epochs)

# Plot values:
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

# Save plot to disk:
plt.savefig(outputPath)
print("[signnet - Train] Saved Loss and Accuracy Plot to: " + outputPath)
print("[signnet - Train] Done CNN training.")
