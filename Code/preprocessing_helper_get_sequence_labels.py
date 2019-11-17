# %% --------------------------------------- Imports -------------------------------------------------------------------
from pathlib import Path

import pandas as pd
from configuration import image_directory, emotion_directory, emotion_map

# %% create list of target classes from all images
filename_matching = '**/*.png'
targets = {}

pathlist = Path(image_directory).glob(filename_matching)
for path in pathlist:
    key = str(path)
    txt_filename = str(path).replace(image_directory, emotion_directory).replace(".png", "_emotion.txt")

    try:
        with open(txt_filename, 'r') as f:
            value = f.readline().strip()
            emotion = emotion_map.get(value[0])
            targets[key] = emotion
    except:
        targets[key] = ""

# %% create dataframe with images metadata
df = pd.DataFrame.from_dict(targets, orient="index", columns=["class"])
df = df.reset_index()
df.columns = ["name", "class"]

df["subject_name"] = df["name"].str.split("/").apply(lambda x: "/".join(x[:-2]))
df["set_name"] = df["name"].str.split("/").apply(lambda x: "/".join(x[:-1]))
df.to_csv("images_all.csv", index=False)

# %% create dataframe with sequence metadata
df = df.groupby(by=["subject_name", "set_name"]).agg({"name": "max", "class": "max"}).reset_index()
df = df.rename(columns={'name': 'peak_image_filename'})
df.to_csv("df_sequences_all.csv", index=False)
