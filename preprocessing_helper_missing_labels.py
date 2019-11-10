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