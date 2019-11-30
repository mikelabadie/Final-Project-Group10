#%% --------------------------------------- Imports --------------------------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from configuration import validation_images_list_filename, model_filename


#%% load model
model_name = model_filename
model = tf.keras.models.load_model(model_name)


#%% evaluate model accuracy on holdout set
test_df = pd.read_csv(validation_images_list_filename)
test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col="name",
    y_col="class",
    target_size=(32,25),
    batch_size=32,
    seed=42,
    class_mode='categorical',
    shuffle=False)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
scoring = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST)
print(model.metrics_names[1],scoring[1])


#%% predictions
predictions = model.predict_generator(generator=test_generator)#, steps=STEP_SIZE_TEST)
predictions_class = np.argmax(predictions,axis=1)
test_df["prediction"] = predictions_class
test_df["actual"] = test_df["class"].replace(test_generator.class_indices)
test_df["Correct"] = test_df["actual"] == test_df["prediction"]


#%% evaluate predictions

# identify classes that were not predicted well
# confusion matrix

# identify subjects that were not predicted well
# can we identify gender/race that we are better or worse at?

# did we identify peak images at higher rate than off-peak?

