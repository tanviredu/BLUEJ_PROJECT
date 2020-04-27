import numpy as np # linear algebra
import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import cv2

import os
print(os.listdir("D:\Food1\dataset"))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import imutils # a simple image utility library



def image_to_feature_vector(image, size=(32, 32)):
# resize the image to a fixed size, then flatten the image into
# a list of raw pixel intensities
    return cv2.resize(image, size).flatten()

# error eikhane... oder train ta jeivabe rakha oivabe rakhte hobe kemne rakhbo?
#usesof datset e konovabe knn ta dye deya jauna?
dataset="D:\Food1\dataset"

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(dataset))
print(len(imagePaths))
print(imagePaths[0])
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = io.imread(imagePath) #cat.0.jpg, cat.1.jpg
    label = imagePath.split(os.path.sep)[-1].split(".")[0] #cat, cat
    gray = rgb2gray(image)
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    
    
    plt.imshow(image, cmap="BrBG")
    plt.axis('off')
    plt.show()
    
    plt.imshow(gray, cmap="gray")
    plt.axis('off')
    plt.show()
       
    plt.imshow(binary, cmap="gray")
    plt.axis('off')
    plt.show()
    

    # extract raw pixel intensity "features", followed by a color
    # histogram to characterize the color distribution of the pixels
    # in the image
    pixels = image_to_feature_vector(image)
    rawImages.append(pixels)  #features of cat.0.jpg, cat.1.jpg
    labels.append(label) #cat, cat, cat
    


# show an update every 1,000 images
#if i > 0 and i % 10000 == 0:
    #print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
labels = np.array(labels)

print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=20)

# Select a subset of the entire dataset
rawImages_subset = rawImages[:2000]
labels_subset= labels[:2000]
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages_subset, labels_subset, test_size=0.25, random_state=20)

# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
neighbors = [1, 3, 5, 7, 9, 13]
for k in neighbors:
    model = KNeighborsClassifier(n_neighbors= k)
    model.fit(trainRI, trainRL)
    acc = model.score(testRI, testRL)
    print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
