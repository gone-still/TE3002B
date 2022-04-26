import numpy as np
import cv2 as cv
import os

# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv.namedWindow(imageName, cv.WINDOW_NORMAL)
    cv.imshow(imageName, inputImage)
    cv.waitKey(0)

def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv.imwrite(imagePath, inputImage, [cv.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)

# image path
path ="D://opencvImages//symbols//samples//"

# nearest neighbors
maxNeighbors = 1

showImages = False
writeResults = True
outPath = "D://opencvImages//classifiers//"

# writeMode: [a]ppend, [w]rite
writeMode = "w"

# Open file for writing:
if writeResults:
    outputFile = open(outPath + "classifierResults.txt", writeMode)
    outputFile.write("i, truth, knn, svm\n")

cellWidth = 210
cellHeight = 165

imageCounter = 0
scale = 1

verbose = False

# the data is stored here:
totalClasses = 14
totalTest = 4

totalSamples = 18

sampleMatrix = np.empty((totalClasses, totalSamples, cellHeight, cellWidth), dtype=np.uint8)

# the class dictionary:
classDictionary = {}
classDictionary[0] = "arrUp"
classDictionary[1] = "faceSmile"
classDictionary[2] = "arrDwn"
classDictionary[3] = "arrZ"
classDictionary[4] = "arrShck"

classDictionary[5] = "arrDouble"
classDictionary[6] = "charH"
classDictionary[7] = "charE"
classDictionary[8] = "charL"
classDictionary[9] = "charO"


classCounter = 0
sampleCounter = 0

# Iterate over the names of each class
directoryList = os.listdir(path)
for currentDirectory in os.listdir(path):

    # Get the directory on the current path:
    currentPath = os.path.join(path, currentDirectory)

    sampleCounter = 0

    # Get the images on the current path:
    for currentImage in os.listdir(currentPath):

        # create path and read image:
        imagePath = os.path.join(currentPath, currentImage)

        inputImage = cv.imread(imagePath)
        imageCopy = inputImage.copy()

        if verbose:
            showImage("Image: " + str(imageCounter), inputImage)

        grayImage = cv.cvtColor(inputImage, cv.COLOR_BGR2GRAY)

        # get dimensions:
        originalHeight, originalWidth = grayImage.shape[:2]

        invertedImage = 255 - grayImage
        imageArea = cv.countNonZero(invertedImage)

        # if verbose:
        #     print("Original Area: " + str(imageArea))

        minArea = 0.05 * imageArea

        # if verbose:
        #    print("minArea: " + str(minArea))

        # Find the big contours/blobs on the filtered image:
        contours, hierarchy = cv.findContours(invertedImage, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        maxX = 0
        minX = originalWidth
        minY = originalHeight
        maxY = 0

        # Look for the outer bounding boxes:
        for b, c in enumerate(contours):

            currentArea = cv.contourArea(c)

            if hierarchy[0][b][3] == -1 and currentArea > minArea:

                x, y, w, h = cv.boundingRect(c)

                #cv.rectangle(imageCopy,(x,y),(x+w,y+h),(0,255,0),1)
                #showImage("Rects", imageCopy)

                contourMaximumX = x + w
                if contourMaximumX > maxX:
                    maxX = contourMaximumX
                if x < minX:
                    minX = x
                contourMaximumY = y + h
                if contourMaximumY > maxY:
                    maxY = contourMaximumY
                if y < minY:
                    minY = y

        compoX = minX
        compoY = minY
        compoWidth = maxX - minX
        compoHeight = maxY - minY

        #cv.rectangle(imageCopy, (compoX, compoY), (compoX + compoWidth, compoY + compoHeight), (255, 0, 0), 1)
        #showImage("Rects", imageCopy)

        # Crop the ROI:
        # filteredImage[y:y + h, x:x + w]
        croppedImg = grayImage[compoY:compoY + compoHeight, compoX:compoX + compoWidth]
        if verbose:
            showImage("cropped img", croppedImg)
            # writeImage("D://opencvImages//sigSamples//data//croppedSig.png", croppedImg)

        # create canvas:
        newImage = np.zeros((cellHeight, cellWidth), np.uint8)
        newImage = 255 - newImage
        croppedHeight, croppedWidth = croppedImg.shape[:2]

        # top-left point from which to insert the smallest image. height first, from the top of the window
        offset = np.array((compoY, compoX))
        newImage[offset[0]:offset[0] + croppedHeight, offset[1]:offset[1] + croppedWidth] = croppedImg

        # Resize?
        if scale != 1:
            width = int(newImage.shape[1] * scale)
            height = int(newImage.shape[0] * scale)

            # dsize
            dsize = (width, height)

            # resize image
            newImage = cv.resize(newImage, dsize, interpolation=cv.INTER_AREA)

        if verbose:
            showImage("newImage", newImage)
            # writeImage(path + "newImage", newImage)

        # Store in array:
        sampleMatrix[classCounter, sampleCounter] = newImage
        # shit = sampleMatrix[0, 0]
        # showImage("shit", shit)
        sampleCounter = sampleCounter + 1
        print("Class: "+str(classCounter)+" Sample: "+str(sampleCounter))

    classCounter = classCounter + 1

# Reshape data to a plain matrix:
train = sampleMatrix[:totalClasses-totalTest, :].reshape(-1, cellWidth*cellHeight).astype(np.float32)
# train = sampleMatrix.reshape(-1, cellWidth*cellHeight).astype(np.float32)
testImages = sampleMatrix[totalClasses-totalTest:, :]
test = testImages.reshape(-1, cellWidth*cellHeight).astype(np.float32)

# Create labels for train and test data
k = np.arange(totalClasses-totalTest)
train_labels = np.repeat(k, totalSamples)[:, np.newaxis]
# test_labels = train_labels.copy()
test_labels = np.array([[0], [1], [3], [2], [9], [9], [7], [7], [8], [8], [0], [2], [2], [3], [3], [4], [4], [1],
                        [1], [5], [8], [8], [1], [2], [2], [2], [5], [5], [1], [6], [6], [7], [7], [5], [9], [9],
                        [1], [0], [0], [0], [2], [3], [3], [4], [4], [5], [1], [1], [6], [6], [6], [5], [4], [3],
                        [5], [4], [0], [6], [8], [5], [0], [2], [1], [7], [8], [9], [1], [2], [3], [8], [0], [1],
                        ])

(testFolders, testSamples) = testImages.shape[:2]

for f in range(testFolders):
    for m in range(testSamples):
        testes = testImages[f][m]

        dictionaryValue = test_labels[totalSamples*f + m][0]

        currentClass = classDictionary[dictionaryValue]
        if verbose:
            print("Smaple: "+str(f)+"-"+str(m)+", class: "+currentClass)
            showImage("teste", testes)

    # print("Dim: "+str(train.shape))


print("Running Knn...")

# Initiate kNN, train it on the training data, then test it with the test data with k=5
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret, knnResult, neighbours, dist = knn.findNearest(test, maxNeighbors)

# if writeResults:
#    outputFile.write("Knn results, k = " + str(maxNeighbors) + "\n")

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = knnResult == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / knnResult.size
print("Knn Acc: "+str(accuracy))

print("Running SVM...")

SVM = cv.ml.SVM_create()
SVM.setKernel(cv.ml.SVM_LINEAR)
SVM.setType(cv.ml.SVM_C_SVC)
SVM.setC(2.0)
SVM.setGamma(5.5)
# SVM.setC(2.67)
# SVM.setGamma(5.383)

SVM.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
SVM.train(train, cv.ml.ROW_SAMPLE, train_labels)

svmResult = SVM.predict(test)[1]

mask = svmResult == test_labels
correct = np.count_nonzero(mask)
print("SVM Acc: "+str(correct*100.0/svmResult.size))

(h, w) = testImages.shape[:2]
labelIndex = 0

for y in range(h):
    currentArray = testImages[y]
    (h, w) = currentArray.shape[:2]
    for x in range(h):
        currentImage = currentArray[x]

        knnPrediction = knnResult[labelIndex][0]
        knnLabel = classDictionary[knnPrediction]

        svmPrediction = svmResult[labelIndex][0]
        svmLabel = classDictionary[svmPrediction]

        dictionaryValue = test_labels[labelIndex][0]
        currentLabel = classDictionary[dictionaryValue]

        writeString = str(labelIndex) + ", " + str(dictionaryValue) + ", " + \
                      str(knnPrediction) + ", " + str(svmPrediction) + ", " + \
                      "(" + currentLabel + ")" + ", [" + knnLabel + "]" + ", [" + svmLabel + "]"

        labelIndex = labelIndex + 1

        print(writeString)
        if True:
            showImage("Test Image", currentImage)

        if writeResults:
            outputFile.write(writeString + "\n")

# Close writing file:
if writeResults:
    outputFile.close()

outString = "HELLO :D"
print(outString)