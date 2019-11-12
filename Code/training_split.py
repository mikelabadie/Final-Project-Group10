import pandas as pd
from sklearn.model_selection import train_test_split
from configuration import training_sample_list_filename, validation_sample_list_filename, \
    percent_images_before_peak_for_training, percent_images_before_peak_for_validation, \
    training_images_list_filename, validation_images_list_filename


#%% merge the given sequence labels and our manual sequence labels into a single csv
df = pd.read_csv("df_sequences_all.csv")

df2 = pd.read_csv("df_sequences_missing_labels_completed.csv")
df2 = df2[["peak_image_filename","class"]]
df2.columns = ["peak_image_filename","class_manual"]
df2 = pd.merge(df, df2, how="left",on="peak_image_filename")
df2["class"] = df2["class"].combine_first(df2["class_manual"])
df2 = df2[['subject_name', 'set_name', 'peak_image_filename', 'class']]


#%% create training and validation datasets by sequence
df = df2.dropna(subset=["class"])
train, validation = train_test_split(df,test_size=0.2)
train.to_csv(training_sample_list_filename,index=False)
validation.to_csv(validation_sample_list_filename,index=False)


#%% training set: get images near peak image
images_all = pd.read_csv("df_images_all.csv")

images_to_use = {}
for _, row in train.iterrows():  #df.dropna(subset=["class"]).iterrows():
    images_for_sequence = list(images_all[images_all["set_name"]==row["set_name"]]["name"].sort_values())
    num_to_use_for_sequence = int(percent_images_before_peak_for_training*len(images_for_sequence))
    images_to_use_for_sequence = images_for_sequence[len(images_for_sequence)-num_to_use_for_sequence:]
    for img in images_to_use_for_sequence:
        images_to_use[img]=row["class"]
images_to_use = pd.DataFrame.from_dict(images_to_use,orient='index',columns=["class"]).sort_index().reset_index()
images_to_use = images_to_use.rename(columns={"index":"name"})
images_to_use.to_csv(training_images_list_filename,index=False)

images_to_use = {}
for _, row in validation.iterrows():  #df.dropna(subset=["class"]).iterrows():
    images_for_sequence = list(images_all[images_all["set_name"]==row["set_name"]]["name"].sort_values())
    num_to_use_for_sequence = int(percent_images_before_peak_for_training*len(images_for_sequence))
    images_to_use_for_sequence = images_for_sequence[len(images_for_sequence)-num_to_use_for_sequence:]
    for img in images_to_use_for_sequence:
        images_to_use[img]=row["class"]
images_to_use = pd.DataFrame.from_dict(images_to_use,orient='index',columns=["class"]).sort_index().reset_index()
images_to_use = images_to_use.rename(columns={"index":"name"})
images_to_use.to_csv(validation_images_list_filename,index=False)