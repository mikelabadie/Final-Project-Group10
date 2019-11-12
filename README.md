# DeepLearningEmotions

## Setup
I run the commands in project_instance_config.txt to update my environment and to download/unzip data files.
1. Unzip images and metadata to local machine
2. Change Code/configuartion.py to reflect locations of unzipped files (though it is probably easier to unzip to match file structure as observed in Code/configuration.py).

## Code
### Preprocessing
This can be skipped if you have Code/df_images_all.csv, Code/df_sequences_all.csv, and Code/df_sequences_missing_labels_completed.csv.  When these are read as dataframes, you will need to update filenames to match your file structure (as discussed above).
1. Code/preprocessing_helper_get_sequence_labels.py:  this gets labels for sequences that have labels.  this also gets a list of all images by its sequence.
2. Code/preprocessing_helper_missing_labels.py:  i used this to create a csv of the peak image for sequences that did not have labels.  i manually tagged the sequences.  

### Training
1. Code/training_split.py:  this gets the csv of sequences (with tags).  assigns the tag for the peak image to all images in the sequence.  i use images that are close to the peak image.  the number/percentage of nearby images is configurable.  this creates csv for the training and testing sets.
2. Code/training_model_building.py:  uses keras imagedatagenerator.flow_from_dataframe for pipeline
