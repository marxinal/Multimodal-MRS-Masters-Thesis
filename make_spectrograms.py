import os
import pandas as pd
import re
import math
import numpy as np
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings

# Remove warnings
warnings.filterwarnings("ignore")

# Establish general path to save the images
path = "/Users/helenk05/Desktop/Final_Folder/spectrogram_images/"

# Load in dataset
all_songs_selected = pd.read_excel("/Users/helenk05/Desktop/Final_Folder/all_songs_selected.xlsx")
# Create a path to spectrogram iamges
all_songs_selected['spectrogram_path'] = ""
pd.options.mode.chained_assignment = None 

# This was ran multiple times to create images of all songs
# Computer was overloading, CPU too small
for i in np.arange(0, len (all_songs_selected)):
    try:
        file_name = all_songs_selected.filename[i]
        song = file_name.replace(".mp3", "")
        # Skipping songs for which the spectrogram already exists
        if os.path.exists(path + song + ".jpg"):
            print(path + song + ".jpg" + "already exists")
        # Generating spectrograms
        else:
            y, sr = librosa.load(all_songs_selected.filepath_cut[i])
            melspectrogram_array = librosa.feature.melspectrogram(y = y, sr = sr, n_mels = 128,fmax = 8000)
            mel = librosa.power_to_db(melspectrogram_array)
            # Length and Width of Spectogram
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = float(mel.shape[1]) / float(100)
            fig_size[1] = float(mel.shape[0]) / float(100)
            plt.rcParams["figure.figsize"] = fig_size
            plt.axis('off')
            plt.axes([0., 0., 1., 1.0], frameon = False, xticks=[], yticks=[])
            librosa.display.specshow(mel, cmap = 'gray_r')
            plt.savefig(path + song + ".jpg", bbox_inches = None, pad_inches = 0)
            all_songs_selected['spectrogram_path'].loc[i] = str(path + song + ".jpg")
            plt.close()
    except Exception as e:
        print(e)


# Get the path to each image per specific row
for i in np.arange(0, len (all_songs_selected)):
    try:
        file_name = all_songs_selected.filename[i]
        song = file_name.replace(".mp3", "")
        if os.path.exists(path+song+".jpg"):
            print(path+song+".jpg" + " _ already exists")
            all_songs_selected['spectrogram_path'].loc[i] = str(path + song[i] +".jpg")
        else:
            continue
    except Exception as e:
        print(e)


# Check how many songs are missing the spectrogram image and remove them
all_songs_selected.groupby(['tag'])['tag'].count()
all_songs_selected_sp = all_songs_selected.replace(r'^\s*$', np.nan, regex=True)
all_songs_selected_sp.dropna()

# Select equal number of songs per genre 
size = 4550      # sample size (maximum of the genre with the lowest number of songs)
replace = False  # without replacement
fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
all_songs_selected_2 = all_songs_selected_sp.groupby('tag', as_index=False).apply(fn)

# Reset index
all_songs_selected_2.reset_index(drop=True, inplace=True)
# Update and save the final dataframe as a .xlsx file
all_songs_selected_2.to_excel("/Users/helenk05/Desktop/Final_Folder/all_songs_selected_2.xlsx", index = False, encoding='utf-8')

