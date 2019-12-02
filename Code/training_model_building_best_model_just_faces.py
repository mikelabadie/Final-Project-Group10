#%% --------------------------------------- Imports --------------------------------------------------------------------
from PIL import Image
import pandas as pd
import numpy as np
from keras.utils import np_utils
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.metrics import Recall
from keras_preprocessing.image import ImageDataGenerator

from configuration import training_images_list_filename, training_images_list_filename_just_faces, \
    validation_images_list_filename, validation_images_list_filename_just_faces, model_filename_just_faces


#%% ---------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)

image_size = (54,72)

#%%
train_df = pd.read_csv(training_images_list_filename_just_faces)

datagen=ImageDataGenerator(rescale=1./255.,
                           validation_split=0.25,
                           horizontal_flip=True
                           #,height_shift_range=1
                           #,width_shift_range=1
                           #,rotation_range=1
                           )

train_generator=datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="name",
    y_col="class",
    subset="training",
    batch_size=256,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=image_size)

valid_generator=datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="name",
    y_col="class",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=image_size)


#%%
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(image_size[0],image_size[1],3), padding='same'))
model.add(Activation('elu'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('tanh'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(Adam(lr=0.001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])

# checkpoints
checkpoint = ModelCheckpoint(model_filename_just_faces, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
#checkpoint = ModelCheckpoint(model_filename, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
callbacks_list = [checkpoint]

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    callbacks=callbacks_list,
                    epochs=30)