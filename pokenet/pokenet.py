# File        :   pokenet.py (The Pokenet CNN model description)
# Version     :   1.0.0
# Description :   Script that builds the Pokenet
#                 Based on: https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/
# Date:       :   Feb 28, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

# Import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


class pokenet:
    @staticmethod
    # The build method accepts four parameters: the image dimensions, depth,
    # and number of classes in the dataset.
    def build(width, height, depth, classes):
        # Build and initialize the model/network:
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()

        # Set the input axis order (channel order):
        inputShape = (height, width, depth)
        chanDim = -1

        # Let's add the first set of layers to the
        # Network: CONV => RELU => BN => POOL

        # First layer, convolution (filtering) with 8
        # kernels, size of (5, 5)
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        # model.add(Conv2D(16, (5, 5), padding="same", input_shape=inputShape))

        # Next layer, the activation layer with a ReLU function:
        model.add(Activation("relu"))

        # Batch normalization applies a transformation that
        # maintains the mean output close to 0 and the output
        # standard deviation close to 1:
        # Normalize the pixel color or "depth":
        model.add(BatchNormalization(axis=chanDim))

        # POOL Layer for dimensional reduction, the input is
        # Max pooled with a kernel of size (2,2)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))

        # Let's add the second set of layers to the
        # Network: CONV => RELU => CONV => RELU => POOL

        # Convolution (filtering) with 16 kernels, size of (3, 3):
        # model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        # Max pooling:
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        # Now, add the third set of layers to the
        # Network: CONV => RELU => CONV => RELU => POOL

        # Convolution (filtering) with 32 kernels, size of (3, 3):
        # model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        # Max pooling:
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        # Two sets of fully connected layers and a softmax classifier

        # Flatten the mat into a vector:
        model.add(Flatten())
        # Implement the fully connected layer with N neurons
        # N is a tunable hyper parameter:
        # model.add(Dense(1024))
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        # Implement the fully connected layer with N neurons
        # N is a tunable hyper parameter:
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        # Finally, the softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
