# GWU Data Science - Deep Learning - Group 10

## Summary
Describe the project

## Setup
1. I run the commands in project_instance_config.txt to update my environment and to download/unzip data files.
   * Updates environment (cuda drivers, tensorflow-gpu, keras)
   * Unzip images and metadata to local machine
2. Change Code/configuartion.py to reflect locations of unzipped files.   Though it is probably easier to unzip to match file structure as observed in Code/configuration.py.

## Code
#### Preprocessing
This can be skipped if you have Code/df_images_all.csv, Code/df_sequences_all.csv, and Code/df_sequences_missing_labels_completed.csv.
1. Code/preprocessing_helper_get_sequence_labels.py:  creates csv of sequences and images with their labels
    * this gets labels for sequences that have labels.
    * this also gets a list of all images by its sequence.
2. Code/preprocessing_helper_missing_labels.py:  this was used to aid our manual tagging effort
    * i used this to create a csv of the peak image for sequences that did not have labels.  
    * moves all peak images for untagged sequences into one folder
    * creates a csv that can be manually updated with assigned tag for sequence
    * i manually tagged peak images for 266 sequences.  

#### Training
1. Code/training_split.py:  creates csv for the training and testing sets
    * this gets the csv of sequences (with tags).  
    * assigns the tag for the peak image to all images in the sequence.  
    * i use images that are close to the peak image.  
    * the number/percentage of nearby images is configurable.  
2. Code/training_model_building.py:  trains neural network
    * uses keras imagedatagenerator.flow_from_dataframe for pipeline
    * When image filenames are read from dataframes, you will need to update filenames to match your file structure (as discussed above).
