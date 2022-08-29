

import os
import re
from PIL import Image
import pandas as pd
import numpy as np


"""
Slice the spectrogram into multiple 128x128 images which will be the input to the
Convolutional Neural Network.
"""
# Create general path to the folder
path = "/Users/helenk05/Desktop/Final_Folder/spectrogram_cut/"

# Load the dataset 

all_songs_selected_2 = pd.read_excel("/Users/helenk05/Desktop/Final_Folder/all_songs_selected_2.xlsx")

# Slice spectrogram images
pd.options.mode.chained_assignment = None 
# Create a path to save the sliced images
all_songs_selected_2["sliced_path"] = ""
counter = 0

for i in np.arange(0, len (all_songs_selected_2)):
    # get the file name (song title) and genre to save the path
    file_name = all_songs_selected_2.filename[i]
    song = file_name.replace(".mp3", "")
    genre = str(all_songs_selected_2['tag'][i])
    img = Image.open(all_songs_selected_2.spectrogram_path[i])
    subsample_size = 128
    width, height = img.size
    number_of_samples = width / subsample_size
    counter = i
    for j in range(int(number_of_samples)): #changed
        start = j*subsample_size
        img_temporary = img.crop((start, 0., start + subsample_size, subsample_size))
        # save image
        img_temporary.save(path + genre + str(counter) + "_" + ".sliced.jpg")
        # save path
        all_songs_selected_2["sliced_path"].loc[i] = str(path + genre + str(counter) + "_" + ".sliced.jpg")

# Save the new validation dataset with the image path
all_songs_selected_2.to_excel("/Users/helenk05/Desktop/Final_Folder/all_songs_selected_2.xlsx")

# Change the name of the files to "genre_number" instead of song title
path = "/Users/helenk05/Desktop/Final_Folder/spectrogram_cut"
for i in np.arange(0, len (all_songs_selected_2)):
    genre_variable = re.search(path + "/(.+?)_.*.jpg",all_songs_selected_2.sliced_path[i]).group(1)
    index = int(re.search (".*(.+?)_.*.jpg", all_songs_selected_2.sliced_path[i]).group(1))
