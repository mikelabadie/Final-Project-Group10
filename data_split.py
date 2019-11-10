import cv2
import os
import gc
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage.transform import rescale


# Resource used to help: https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9

# This script generates the training set (the one you will be provided with),
# and the held out set (the one we will use to test your model towards the leaderboard).

os.chdir(r'Final_Project')

angry_pictures = 'Data/Angry'
disgust_pictures = 'Data/Disgust'
fear_pictures = 'Data/Fear'
happy_pictures = 'Data/Happy'
sad_pictures = 'Data/Sadness'
surprise_pictures = 'Data/Surprise'

# Define pic parameters (original pixel sizes)
nrows = 640  # 640 pixels by
ncolumns = 480  # 480 pixels
channels = 2  # using greyscale

processed_pics = []
labels = []

# Process Angry Pictures
angry_list = os.listdir(angry_pictures)
for angry_pic in angry_list:
    path = os.path.join(angry_pictures, angry_pic)
    path = os.path.abspath(path)

    # Process original image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC)
    processed_pics.append(img)
    label = 0  # 'Angry'
    labels.append(label)

# Process Disgust Pictures
disgust_list = os.listdir(disgust_pictures)
for disgust_pic in disgust_list:
    path = os.path.join(disgust_pictures, disgust_pic)
    path = os.path.abspath(path)

    # Process original image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC)
    processed_pics.append(img)
    label = 1  # 'Disgust'
    labels.append(label)

# Process Fear Pictures
fear_list = os.listdir(fear_pictures)
for fear_pic in fear_list:
    path = os.path.join(fear_pictures, fear_pic)
    path = os.path.abspath(path)

    # Process original image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC)
    processed_pics.append(img)
    label = 2  # 'Fear'
    labels.append(label)

# Process Happy Pictures
happy_list = os.listdir(happy_pictures)
for happy_pic in happy_list:
    path = os.path.join(happy_pictures, happy_pic)
    path = os.path.abspath(path)

    # Process original image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC)
    processed_pics.append(img)
    label = 3  # 'Happy'
    labels.append(label)

# Process Sad Pictures
sad_list = os.listdir(sad_pictures)
for sad_pic in sad_list:
    path = os.path.join(sad_pictures, sad_pic)
    path = os.path.abspath(path)

    # Process original image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC)
    processed_pics.append(img)
    label = 4  # 'Sad'
    labels.append(label)

# Process Surprise Pictures
surprise_list = os.listdir(surprise_pictures)
for surprise_pic in surprise_list:
    path = os.path.join(surprise_pictures, surprise_pic)
    path = os.path.abspath(path)

    # Process original image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC)
    processed_pics.append(img)
    label = 5  # 'Surprise'
    labels.append(label)


X = np.array(processed_pics)
y = np.array(labels)

print("Shape of images is: ", X.shape)
print("Shape of labels is: ", y.shape)

SEED = 666
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=SEED, test_size=0.2, stratify=y)

print("Shape of train images is: ", x_train.shape)
print("Shape of test images is: ", x_test.shape)
print("Shape of train labels is: ", y_train.shape)
print("Shape of test labels is: ", y_test.shape)

del X
del y
gc.collect()

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)
