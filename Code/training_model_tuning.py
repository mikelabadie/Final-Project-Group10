# %% --------------------------------------- Imports --------------------------------------------------------------------
import os
import random

import numpy as np
import pandas as pd
import talos
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.initializers import glorot_uniform
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam, SGD, Nadam, RMSprop
from keras_preprocessing.image import ImageDataGenerator
from talos.model.early_stopper import early_stopper
from talos.model.normalizers import lr_normalizer

from configuration import training_images_list_filename_just_faces, validation_images_list_filename_just_faces

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
p = {'lr': (0.0001, 0.001, 0.01),
     'neurons_layer_2': [32, 64, 128],
     'neurons_layer_3': [32, 64, 128, 256],
     'neurons_layer_4': [32, 64, 128, 256],
     'neurons_layer_5': [64, 128, 256, 512],
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
     'activation_6': ['relu', 'elu', 'tanh'],
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
    model.add(Conv2D(params['neurons_layer_4'], (3, 3), padding='same'))
    model.add(Activation(params['activation_4']))
    model.add(Conv2D(params['neurons_layer_5'], (3, 3)))
    model.add(Activation(params['activation_5']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout']))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation(params['activation_6']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(8, activation=params['last_activation']))
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
    from itertools import izip

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
results_df.to_csv(r'tuning_results_NEW.csv')

# %% ------------------------------------------ Validate Best Model ----------------------------------------------------
# Get the best model from the results and try below:

# %%
# STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
# history = model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_TEST)
