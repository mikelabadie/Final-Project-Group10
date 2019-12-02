# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import os
import cv2
import pandas as pd


def predict(x_test):
    # %% --------------------------------------------- Data Prep -------------------------------------------------------
    nrows = 72
    ncolumns = 54

    processed_pics = []

    for _, picture in x_test.iterrows():
        picture_path = os.path.abspath(picture["name"])
        processed_pics.append(cv2.resize(cv2.imread(picture_path, cv2.IMREAD_COLOR), (nrows, ncolumns),
                                         interpolation=cv2.INTER_AREA))

    x = np.array(processed_pics)
    x = x.astype('float16')
    print("Shape of images is: ", x.shape)

    # %% --------------------------------------------- Predict ---------------------------------------------------------
    model = tf.keras.models.load_model('model_just_faces.hdf5')
    # If using more than one model to get y_pred, they need to be named as "mlp_ajafari1.hdf5", ""mlp_ajafari2.hdf5",
    # etc.
    y_pred = np.argmax(model.predict(x), axis=1)
    return y_pred, model


x_test = pd.read_csv('original_test_list.csv')
preds, model = predict(x_test)

x_test['Predicted'] = preds
for _, row in x_test.iterrows():
    pred = row['Predicted']

    if pred == 0:
        x_test['Predicted'][_] = 'anger'
    if pred == 1:
        x_test['Predicted'][_] = 'contempt'
    if pred == 2:
        x_test['Predicted'][_] = 'disgust'
    if pred == 3:
        x_test['Predicted'][_] = 'fear'
    if pred == 4:
        x_test['Predicted'][_] = 'happy'
    if pred == 5:
        x_test['Predicted'][_] = 'sadness'
    if pred == 6:
        x_test['Predicted'][_] = 'surprise'

print(x_test)
