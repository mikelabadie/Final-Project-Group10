# DeepLearningEmotions

Setup
1. Unzip images and metadata to local machine (see Machine Setup)
2. Change project_configuartion.py to reflect locations of unzipped files

Preprocessing
This can be skipped if you have images_all.csv, sequences_all.csv, and sequences_missing_labels_completed.csv.  When these are read as dataframes, you will need to update filenames to match your file structure.
1. preprocessing_helper_get_sequence_labels.py:  this gets labels for sequences that have labels.  this also gets a list of all images by its sequence.
2. preprocessing_helper_missing_labels.py:  i used this to create a csv of the peak image for sequences that did not have labels.  i manually tagged the sequences.  

Training
1. training_split.py:  this gets the csv of sequences (with tags).  assigns the tag for the peak image to all images in the sequence.  i use images that are close to the peak image.  the number/percentage of nearby images is configurable.  this creates csv for the training and testing sets.
2.  training_model_building.py:  uses keras imagedatagenerator.flow_from_dataframe for pipeline
