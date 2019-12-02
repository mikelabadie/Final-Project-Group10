## Preprocessing
These scripts were used to create datasets and metadata for training our models. 
* preprocessing_get_sequence_labels.py
* training_split.py
* preprocessing_helper_face_detection.py

## Modeling
These scripts represent the progression of our modeling efforts.
  * training_model_building.py
  * training_model_building_tuning.py
  * training_model_building_best_model.py
  * training_model_building_best_model_just_faces.py
  
## Evaluation
These scripts were used to evaluate the quality of our trained models.
* model_evaluation.py
* model_evaluation_helpers.py

## CSV Used in Modeling
These csv files were used to contain links and class information for our training and testing sets.  
* Data Inventory: This is the output of preprocessing_get_sequence_labels.py and our manual tagging.  It is the main effort at taking inventory of our dataset.  The output is used by training_split.py and preprocessing_helper_face_detection.py.
  * images_all.csv
  * sequences_all.csv
  * sequences_missing_labels_completed.csv
* Training Set:  This is the output of training_split.py and preprocessing_helper_face_detection.py.  These are used to build the dataframes on which our models are trained and evaluated.
  * images_training_list.csv
  * images_validation_list.cv
  * images_training_list_just_faces.csv
  * images_validation_list_just_faces.csv
* Tuning Results:  This is the output of training_model_building_tuning.py.  We used these results to tune our models.
  * tuning_results_2.csv
  * tuning_results_3.csv
