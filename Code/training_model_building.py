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

from configuration import image_directory, augmented_image_directory, \
    training_images_list_filename, training_augmented_sample_list_filename, \
    validation_images_list_filename, \
    class_map, num_classes, model_filename, \
    LR, N_NEURONS, N_EPOCHS, BATCH_SIZE, DROPOUT, IMAGE_SIZE, \
    resize_image


#%% ---------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)


#%%
train_df = pd.read_csv(training_images_list_filename)
test_df = pd.read_csv(validation_images_list_filename)

datagen=ImageDataGenerator(rescale=1./255., validation_split=0.25)

train_generator=datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="name",
    y_col="class",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(32,25))

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
    target_size=(32,25))

test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col="name",
    y_col="class",
    target_size=(32,25),
    batch_size=32,
    seed=42,
    class_mode='categorical')


#%%
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same'))#, input_shape=(64,49)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.compile(RMSprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10)


#%%
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.evaluate_generator(generator=valid_generator,steps=STEP_SIZE_TEST)