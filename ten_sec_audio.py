from pydub import AudioSegment
import glob
import numpy as np
import pandas as pd
import os 

"""
Cutting audio again for 10-seconds to be used in the experiment
"""
# Loading the 30-second audios
cut_filez = glob.glob("/Users/helenk05/Desktop/Final_Folder/all_songs_cut/*.mp3")  
song_path = ''
# Creating general path to save the songs
path = '/Users/helenk05/Desktop/Final_Folder/ten_second_songs/'

for i in np.arange(0, len (cut_filez)):
    try:
        #importing file from location by giving its path
        sound = AudioSegment.from_mp3(cut_filez[i])
        #Selecting Portion we want to cut
        StrtMin = 0
        StrtSec = 0
        EndMin = 0
        EndSec = 10
        # Time to milliseconds conversion
        StrtTime = StrtMin*60*1000+StrtSec*1000
        EndTime = EndMin*60*1000+EndSec*1000
        # Opening file and extracting portion of it
        extract = sound[StrtTime:EndTime]
        song_path = str(os.path.basename(path + cut_filez[i]))
        f_name = song_path.replace('_cut.mp3', '')
        # Saving file in required location
        extract.export(path + f_name + "_cut10.mp3", format="mp3")
    except Exception as e:
        print(e)

