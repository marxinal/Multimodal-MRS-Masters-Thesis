from pydub import AudioSegment
import glob
import numpy as np
import pandas as pd
import os 

# Load in the dataset
all_songs_selected = pd.read_excel("/Users/helenk05/Desktop/Final_Folder/all_songs_selected.xlsx")

# Establish general path to save the cut audio
path = "/Users/helenk05/Desktop/Final_Folder/all_songs_cut/"
all_songs_selected['filepath_cut'] = ''
pd.options.mode.chained_assignment = None 

all_songs_selected["filename"] = ""

for i in np.arange(0, len (all_songs_selected)):
    try:
        # Importing file from location by giving its path
        sound = AudioSegment.from_mp3(all_songs_selected.filepath_name[i])
        # Selecting Portion we want to cut
        StrtMin = 1
        StrtSec = 10
        EndMin = 1
        EndSec = 40
        # Time to milliseconds conversion
        StrtTime = StrtMin*60*1000+StrtSec*1000
        EndTime = EndMin*60*1000+EndSec*1000
        # Opening file and extracting portion of it
        extract = sound[StrtTime:EndTime]
        # Saving the path for easier retrieval of the songs
        all_songs_selected["filename"].loc[i] = str(os.path.basename(all_songs_selected["filepath_name"][i]))
        f_name, f_ext = os.path.splitext(all_songs_selected.filename[i])
        # Saving file in required location
        extract.export(path + f_name + "_cut.mp3", format="mp3")
        all_songs_selected['filepath_cut'].loc[i] = str(path + f_name + "_cut.mp3")
    except Exception as e:
        print(e)

# Removing songs for which audio was not cut
all_songs_selected  = all_songs_selected.replace(r'^\s*$', np.nan, regex=True)
all_songs_selected = all_songs_selected.dropna() 

# Saving the paths to the 30-second songs
all_songs_selected.to_excel("/Users/helenk05/Desktop/Final_Folder/all_songs_selected.xlsx", index = False, encoding='utf-8')

