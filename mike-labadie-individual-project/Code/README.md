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
  * <b>training_model_building_best_model.py</b>:  implementation of the best model found in tuning.
  * <b>training_model_building_best_model_just_faces.py</b>:  implementation of the best model, trained on the face extract images.
  
## Evaluation
These scripts were used to evaluate the quality of our trained models.
* <b>model_evaluation.py</b>:  evaluates the three models on the hold out set.
* <b>model_evaluation_helpers_mlabadie.py</b>:  implements a confusion matrix as given by sklearn.
