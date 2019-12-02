This is a description of our coding products for this project.  The order represents the logical order needed to replicate our work.

## Configuration
This file represents a centralized area for settings (i.e. file paths, naming, etc) used throughout the project.
* <b>configuration.py</b>:  contains settings (i.e file paths, naming, etc) used throughout the project.

## Preprocessing
These scripts were used to create datasets and metadata for training our models. 
* <b>preprocessing_get_sequence_labels.py</b>: creates initial datasets built by finding all images and their associated class labels.
* <b>training_split.py</b>:  splits the dataset into a training and testing set.
* <b>preprocessing_helper_face_detection.py</b>:  creates new images with extracts of just the subject's face.

## Modeling
These scripts represent the progression of our modeling efforts.
  * <b>training_model_building.py</b>:  first model, as taken from Keras's website.
  * <b>training_model_tuning.py</b>:  performs hyperparameter tuning.
  * <b>training_model_building_best_model.py</b>:  implementation of the best model found in tuning.
  * <b>training_model_building_best_model_just_faces.py</b>:  implementation of the best model, trained on the face extract images.
  
## Evaluation
These scripts were used to evaluate the quality of our trained models.
* <b>model_evaluation.py</b>:  evaluates the three models on the hold out set.
* <b>model_evaluation_helpers_mlabadie.py</b>:  implements a confusion matrix as given by sklearn.

## CSV Used in Modeling
These csv files were used to contain links and class labels for our training and testing image sets.  
* Data Inventory: This is the output of preprocessing_get_sequence_labels.py and our manual tagging.  It is the main effort at taking inventory of our dataset.  The output is used by training_split.py and preprocessing_helper_face_detection.py.
  * images_all.csv
  * sequences_all.csv
  * sequences_missing_labels_completed.csv
* Training Set:  This is the output of training_split.py and preprocessing_helper_face_detection.py.  These are used to build the dataframes on which our models are trained and evaluated.
  * images_training_list.csv
  * images_validation_list.csv
  * images_training_list_just_faces.csv
  * images_validation_list_just_faces.csv
* Tuning Results:  This is the output of training_model_building_tuning.py.  We used these results to tune our models.
  * tuning_results_2.csv
  * tuning_results_3.csv
