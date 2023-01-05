# %% --------------------------------------- Imports --------------------------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from configuration import validation_images_list_filename, validation_images_list_filename_just_faces, \
    model_filename, model_filename_first, model_filename_just_faces, class_map

from model_evaluation_helpers_mlabadie import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

# %% evaluate model accuracy of keras example model on holdout set without facial extraction
image_size = (32, 25)  # no facial extract
model_name = model_filename_first
validation_list = validation_images_list_filename_just_faces

test_df = pd.read_csv(validation_list)
test_datagen = ImageDataGenerator(rescale=1. / 255.)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col="name",
    y_col="class",
    target_size=image_size,
    batch_size=32,
    seed=42,
    class_mode='categorical',
    shuffle=False)

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
model = tf.keras.models.load_model(model_name)
scoring = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST, verbose=0)
print("Keras CNN Example", model.metrics_names[1], scoring[1])

# %% evaluate model accuracy of tuned model on holdout set without facial extraction
image_size = (32, 25)  # no facial extract
model_name = model_filename_just_faces
validation_list = validation_images_list_filename

test_df = pd.read_csv(validation_list)
test_datagen = ImageDataGenerator(rescale=1. / 255.)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col="name",
    y_col="class",
    target_size=image_size,
    batch_size=32,
    seed=42,
    class_mode='categorical',
    shuffle=False)

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
model = tf.keras.models.load_model(model_name)
scoring = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST, verbose=0)
print("No facial extraction", model.metrics_names[1], scoring[1])

# %% evaluate model accuracy on holdout set with facial extraction
image_size = (54, 72)  # facial extract
model_name = model_filename_just_faces
validation_list = validation_images_list_filename_just_faces

test_df = pd.read_csv(validation_list)
test_datagen = ImageDataGenerator(rescale=1. / 255.)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col="name",
    y_col="class",
    target_size=image_size,
    batch_size=32,
    seed=42,
    class_mode='categorical',
    shuffle=False, verbose=0)

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
model = tf.keras.models.load_model(model_name)
scoring = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST, verbose=0)
print("With facial extraction", model.metrics_names[1], scoring[1])

# %% predictions
predictions = model.predict_generator(generator=test_generator)  # , steps=STEP_SIZE_TEST)
predictions_class = np.argmax(predictions, axis=1)
test_df["prediction"] = predictions_class
inv_map = {v: k for k, v in test_generator.class_indices.items()}
test_df["prediction_class"] = test_df["prediction"].replace(inv_map)

test_df["actual"] = test_df["class"].replace(test_generator.class_indices)
test_df["Correct"] = test_df["class"] == test_df["prediction_class"]

# %% evaluate predictions

# identify classes that were not predicted well
# confusion matrix
actuals_classes, predictions_classes = test_df["actual"].values, test_df["prediction"].values
classes_list = np.array(sorted(list(test_generator.class_indices.keys())))
plot_confusion_matrix(actuals_classes, predictions_classes, classes=classes_list,
                      normalize=True,
                      title='Confusion Matrix')
plt.show()

print("Classification Report")
print(classification_report(test_df["class"], test_df["prediction_class"], np.unique(test_df["class"].values)))

# identify subjects that were not predicted well
# can we identify gender/race that we are better or worse at?

# did we identify peak images at higher rate than off-peak?
