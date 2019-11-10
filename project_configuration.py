image_directory="/home/ubuntu/cohn-kanade-images"
emotion_directory="/home/ubuntu/Emotion"
facs_directory="/home/ubuntu/FACS"
landmarks_directory="/home/ubuntu/Landmarks"
emotion_map={"0":"neutral","1":"anger","2":"contempt","3":"disgust","4":"fear","5":"happy","6":"sadness","7":"surprise"}

from PIL import Image
import numpy as np

def resize_image(filename, target_size):
    im = Image.open(filename)
    old_im = im

    desired_size=target_size
    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))

    new_im_array = np.asarray(new_im)
    return new_im, new_im_array, old_im
