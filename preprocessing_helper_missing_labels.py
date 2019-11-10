import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from project_configuration import image_directory
from shutil import copyfile

df = pd.read_csv("labels.csv")

count=0
for _, row in df[df["class"].isna()].iterrows():
    new_filename="/home/ubuntu/Project/NoLabel/"+row["peak_image_filename"].split("/")[-1]
    copyfile(row["peak_image_filename"],new_filename)
    count+=1


#%%
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
#df2["already_tagged"]=labels_by_subject
df2["missing_tags"]=missing_labels_by_subject


#%%
df3 = pd.merge(df,df2,how="left")
df3 = df3[df3["class"].isna()]
df3.to_csv("missing_labels.csv",index=False)