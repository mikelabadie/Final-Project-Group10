# %% --------------------------------------- Imports --------------------------------------------------------------------
from PIL import Image
import pandas as pd
import numpy as np
from keras.utils import np_utils
import os
import talos
import random
import numpy as np
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD, Nadam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.metrics import Recall
from keras_preprocessing.image import ImageDataGenerator

from configuration import image_directory, augmented_image_directory, \
    training_images_list_filename, training_augmented_sample_list_filename, \
    validation_images_list_filename, \
    class_map, num_classes, model_filename, \
    LR, N_NEURONS, N_EPOCHS, BATCH_SIZE, DROPOUT, image_size, \
    resize_image, training_images_list_filename_just_faces, validation_images_list_filename_just_faces

from talos.model.normalizers import lr_normalizer
from talos.model.early_stopper import early_stopper
from talos import Evaluate

# %% ---------------------------------------- Set-Up -------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)

image_size = (54, 72)

# %%
# train_df = pd.read_csv(training_images_list_filename)
train_df = pd.read_csv(training_images_list_filename_just_faces)
# test_df = pd.read_csv(validation_images_list_filename)
test_df = pd.read_csv(validation_images_list_filename_just_faces)

datagen = ImageDataGenerator(rescale=1. / 255.,
                             validation_split=0.25,
                             brightness_range=[0.2, 1.0],
                             rotation_range=90,
                             height_shift_range=0.5,
                             horizontal_flip=True)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="name",
    y_col="class",
    subset="training",
    batch_size=32,
    seed=SEED,
    shuffle=True,
    class_mode="categorical",
    target_size=image_size)

valid_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="name",
    y_col="class",
    subset="validation",
    batch_size=32,
    seed=SEED,
    shuffle=True,
    class_mode="categorical",
    target_size=image_size)

test_datagen = ImageDataGenerator(rescale=1. / 255.)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col="name",
    y_col="class",
    target_size=image_size,
    batch_size=32,
    seed=SEED,
    class_mode='categorical')

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
p = {'lr': (0.0001, 10, 10),
     'neurons_layer_2': [32, 64, 128],
     'neurons_layer_3': [32, 64, 128, 256],
     'neurons_layer_4': [32, 64, 128, 256],
     'batch_size': [256, 512],
     'epochs': [30],
     'dropout': (0, 0.50, 10),
     'kernel_initializer': ['uniform', 'normal', 'random_uniform'],
     'weight_regulizer': [None],
     'emb_output_dims': [None],
     'optimizer': [Adam, Nadam, RMSprop, SGD],
     'loss': ['categorical_crossentropy'],
     'activation_1': ['relu', 'elu', 'tanh'],
     'activation_2': ['relu', 'elu', 'tanh'],
     'activation_3': ['relu', 'elu', 'tanh'],
     'activation_4': ['relu', 'elu', 'tanh'],
     'activation_5': ['relu', 'elu', 'tanh'],
     'last_activation': ['softmax']}


# %% -------------------------------------- MLP Tuning ----------------------------------------------------------

def emotions_model(dummyXtrain, dummyYtrain, dummyXval, dummyYval, params):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer=params['kernel_initializer']))
    model.add(Activation(params['activation_1']))
    model.add(Conv2D(params['neurons_layer_2'], (3, 3)))
    model.add(Activation(params['activation_2']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout']))
    model.add(Conv2D(params['neurons_layer_3'], (3, 3), padding='same'))
    model.add(Activation(params['activation_3']))
    model.add(Conv2D(params['neurons_layer_4'], (3, 3)))
    model.add(Activation(params['activation_4']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout']))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation(params['activation_5']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(7, activation=params['last_activation']))
    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['loss'], metrics=['accuracy'])

    history = model.fit(dummyXtrain, dummyYtrain, batch_size=params['batch_size'], epochs=params['epochs'],
                        validation_data=(dummyXval, dummyYval),
                        callbacks=[ModelCheckpoint("conv2d_mwilchek.hdf5", monitor="val_loss", save_best_only=True),
                                   early_stopper(params['epochs'], mode='strict')])

    return history, model


# %% ------------------------------------------ MLP Tuning Eval --------------------------------------------------------
from platform import python_version_tuple

if python_version_tuple()[0] == '3':
    xrange = range
    izip = zip
    imap = map
else:
    from itertools import izip, imap

import numpy as np

tempX, tempY = izip(*(train_generator[i] for i in xrange(len(train_generator))))
trainX, trainY = np.vstack(tempX), np.vstack(tempY)
del tempX, tempY

tempX, tempY = izip(*(valid_generator[i] for i in xrange(len(valid_generator))))
testX, testY = np.vstack(tempX), np.vstack(tempY)
del tempX, tempY

# dummyX, dummyY = train_generator.__getitem__(0)
# testX, testY = valid_generator.__getitem__(0)
# valid_generator.on_epoch_end()

t = talos.Scan(x=trainX,
               y=trainY,
               x_val=testX,
               y_val=testY,
               model=emotions_model,
               params=p,
               experiment_name='emotional_classification',
               round_limit=100)  # just does 10 rounds of modeling / 10 different param configs
# fraction_limit=.005)  # just does 10% of total number param configs)

results = talos.Evaluate(t)
results_df = results.data
results_df = results_df.sort_values(by='val_accuracy', ascending=True)
results_df.to_csv(r'/home/ubuntu/Desktop-Sync-Folder/Check2/tuning_results_4.csv')

# %% ------------------------------------------ Validate Best Model ----------------------------------------------------
# Get the best model from the results and try below:

# %%
# STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
# history = model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_TEST)
