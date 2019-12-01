import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from configuration import image_directory
from shutil import copyfile
from mtcnn.mtcnn import MTCNN
import cv2


#%% facial detection
df = pd.read_csv("images_training_list.csv")

for _, row in df.iterrows():
    # get image
    filename = row["name"]
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # get face from image
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) #mtcnn uses rgb
    detector = MTCNN()
    faces = detector.detect_faces(color_img)
    face = faces[0]

    # extract the bounding box from the requested face
    x1, y1, width, height = face['box']
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face_boundary = image[y1:y2, x1:x2]

    # new image with just the face
    new_filename = filename.replace("cohn-kanade-images","cohn-kanade-images-just-faces")
    print("training",new_filename)
    cv2.imwrite(new_filename, face_boundary)


#%%
df = pd.read_csv("images_validation_list.csv")

for _, row in df.iterrows():
    # get image
    filename = row["name"]
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # get face from image
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) #mtcnn uses rgb
    detector = MTCNN()
    faces = detector.detect_faces(color_img)
    face = faces[0]

    # extract the bounding box from the requested face
    x1, y1, width, height = face['box']
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face_boundary = image[y1:y2, x1:x2]

    # new image with just the face
    new_filename = filename.replace("cohn-kanade-images","cohn-kanade-images-just-faces")
    print("validation",new_filename)
    cv2.imwrite(new_filename, face_boundary)