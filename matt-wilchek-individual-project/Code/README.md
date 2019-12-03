This is a description of our coding products for this project.  The order represents the logical order needed to replicate our work.

## Preprocessing
These scripts were used to create datasets and metadata for training our models. 
* <b>data_split.py</b>:  processes manually reviewed images and splits the dataset into a training and testing set.

## Modeling
These scripts represent the progression of our modeling efforts.
  * <b>training_model_tuning.py</b>:  Primary hyper-parameter tuning done on the CNN model to refine the following 2 scripts
  * <b>training_model_building_best_model.py</b>:  implementation of the best model found in tuning.
  * <b>training_model_building_best_model_just_faces.py</b>:  implementation of the best model, trained on the face extract images.
  
## Evaluation
These scripts were used to evaluate the quality of our trained models.
* <b>model_predict.py</b>:  evaluates the best model on the personal image data collected from family/friends.
