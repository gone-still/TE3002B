# File        :   main.py (FFNN for Digit Recognition)
# Version     :   1.0.0
# Description :   Script that implements a non-ConvNet FFNN for
#             :   digit recognition using keras and MNIST
# Date:       :   Jan 25, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("Tensorflow version: " + str(tf.__version__))

# Total number classes:
classes = 10  # Numbers 0-9
sampleSize = (28, 28)  # size of the images

# Import the MNIST dataset:
# Images of 28 x 28 hand written digits 0-9:
mnist = tf.keras.datasets.mnist

# Unpack the dataset:
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
s = x_train[0].copy()

# Show a sample with black/white color map:
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

# Normalize/Scale the dataset:
# x_train = x_train.astype(np.float) / 255.0
# x_test = x_test.astype(np.float) / 255.0

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Show a scaled sample with black/white color map:
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

# Build the model, it is a sequential model (a FFNN):
model = tf.keras.Sequential()

# Build the FFNN layer by layer:
# Input layer:
model.add(tf.keras.layers.Flatten(input_shape=sampleSize))

# Hidden layers:
# Add a layer with 128 neurons:
model.add(tf.keras.layers.Dense(128))
# Add the activation layer:
model.add(tf.keras.layers.Activation("relu"))

# Output layer:
# The output is one of 10 possible classes:
model.add(tf.keras.layers.Dense(classes))
# Add the activation layer:
model.add(tf.keras.layers.Activation("softmax"))

# Train the model:
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10)

# Assess the model:
val_loss, val_acc = model.evaluate(x_test, y_test)
print((val_loss, val_acc))

# Save the model:
modelFilename = "numberRecognizer.model"
model.save(modelFilename)

# Load model:
numberRecognizer = tf.keras.models.load_model(modelFilename)

# Classify samples:
predictions = numberRecognizer.predict(x_test)

# Print the predictions:
print(predictions)

# Review some classifications:
testClasses = 10  # len(predictions)

for i in range(testClasses):
    # Get the largest prediction:
    largestPrediction = np.argmax(predictions[i])

    # Print it:
    print("FFNN says: " + str(largestPrediction))

    # Show the classified sample:
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.show()
