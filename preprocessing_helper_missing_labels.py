import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from project_configuration import image_directory
from shutil import copyfile


#%% create a csv with sequences that are not tagged
# we used this csv to perform manual tagging on the peak image for 266 sequences

df = pd.read_csv("labels.csv")

# we first copied the peak image for all unlabeled sequences to a single folder for easy tagging
count=0
for _, row in df[df["class"].isna()].iterrows():
    new_filename="/home/ubuntu/Project/NoLabel/"+row["peak_image_filename"].split("/")[-1]
    copyfile(row["peak_image_filename"],new_filename)
    count+=1

# we created a csv that we used to manually code the emotion tag
subjects = []
labels_by_subject = []
missing_labels_by_subject = []

possible_labels = set(df[~(df["class"].isna())]["class"].drop_duplicates())

for subj in df["subject_name"].drop_duplicates():
    already_tagged = set(df[(df["subject_name"]==subj)&~(df["class"].isna())]["class"].values)
    subjects.append(subj)
    labels_by_subject.append(",".join(list(already_tagged)))
    missing_labels_by_subject.append(",".join(list(possible_labels-already_tagged)))

df2 = pd.DataFrame(subjects,columns=["subject_name"])
df2["missing_tags"]=missing_labels_by_subject

df2 = pd.merge(df,df2,how="left")
df2 = df2[df2["class"].isna()]
df2.to_csv("missing_labels.csv",index=False)


#%% merge the given labels and our manual labels into a single csv
df = pd.read_csv("labels.csv")
df2 = pd.read_csv("missing_labels_completed.csv")
df = pd.merge(df,df2[["peak_image_filename","class"]], how="left",on="peak_image_filename")
df.columns = ['subject_name', 'set_name', 'peak_image_filename', 'class','class_manual']
df.to_csv("labels_all.csv",index=False)