import csv
import glob
import os
import time
import pandas as pd

# Searching pattern inside folders and sub folders recursively
# search all jpg files
pattern = r"F:\Matt\PhD Classes\ECE5554_Computer_Vision\Data\CK_validation_MTCNN\**\*.png"
for item in glob.iglob(pattern, recursive=True):
    # delete file
    print("Deleting:", item)
    os.remove(item)

os.chdir(r'F:\Matt\PhD Classes\ECE5554_Computer_Vision\Data\dataset_updated')
directory = "Surprise"
label = "Surprise"

i = 0
files = os.listdir(directory)
fields = ('name', 'class')
for item in files:
    data_file = open('images_MTCNN.csv', 'a')
    name = "/Surprise/" + item
    wr = csv.DictWriter(data_file, fieldnames=fields, lineterminator='\n')
    wr.writerow({'name': name, 'class': label})
    time.sleep(1)
    data_file.close()
    print("Writing file: " + str(i))
    i += 1

df = pd.read_csv(r'images_MTCNN.csv', index_col=False)
df.shape
