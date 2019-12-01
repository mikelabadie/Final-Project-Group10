image_directory="/home/ubuntu/cohn-kanade-images"
augmented_image_directory=""
emotion_directory="/home/ubuntu/Emotion"
facs_directory="/home/ubuntu/FACS"
landmarks_directory="/home/ubuntu/Landmarks"
emotion_map={"1":"anger","2":"contempt","3":"disgust","4":"fear","5":"happy","6":"sadness","7":"surprise"}
class_map = {"anger":1,"contempt":2,"disgust":3,"fear":4,"happy":5,"sadness":6,"surprise":7}
num_classes = 7


#%%
training_sample_list_filename = "sequences_training_list.csv"
validation_sample_list_filename = "sequences_validation_list.csv"
training_images_list_filename = "images_training_list.csv"
validation_images_list_filename = "images_validation_list.csv"
training_augmented_sample_list_filename = ".csv"
percent_images_before_peak_for_training = 0.25
percent_images_before_peak_for_validation = 0.1


#%% ------------------------------------ Modeling ----------------------------------------------------------------------
LR = 1e-3
#N_NEURONS = (512, 256, 256)
N_NEURONS = (1024, 512, 512)
#N_NEURONS = (1024, 512, 512, 512)
N_EPOCHS = 20
BATCH_SIZE = 512 #large enough to ensure it gets good amount of minority classes
DROPOUT = 0.2
IMAGE_SIZE = (64,49) #15,25
model_filename="model.hdf5"


#%%
import cv2
import numpy as np

def resize_image(filename, target_size):
    # im = Image.open(filename)
    # old_im = im

    # desired_size=target_size
    # old_size = im.size  # old_size[0] is in (width, height) format
    #
    # target_ratio = float(desired_size)/max(old_size)
    # new_size = tuple([int(x*target_ratio) for x in old_size])
    # im = im.resize(target_size, Image.ANTIALIAS)
    #
    # # create a new image and paste the resized on it
    # new_im = Image.new("RGB", target_size)
    # new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
    #
    # new_im_array = np.asarray(new_im)

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)  # AREA is better for shrinking images 
    return img
