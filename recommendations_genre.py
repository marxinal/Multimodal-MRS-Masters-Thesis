from numpy import load
from random import sample
import numpy as np
import pandas as pd

# Load test data
test_data = pd.read_excel("/Users/helenk05/Desktop/Final_Folder/test_data.xlsx")

# Get the test set of images created in Google Colab
test_x = load('/Users/helenk05/Desktop/Final_Folder/test_images.npz')
test_x = test_x['arr_0']
images_test = np.array(test_x)

# Load the names of the songs created in Google Colab
song_name = load('/Users/helenk05/Desktop/Final_Folder/song_names.npz')
song_name = song_name['arr_0']


# Create an empty dataframe to store the genre recommendations
column_names = ['anchor_song', 'recommendation']
genre_recommendations = pd.DataFrame(columns = column_names)

# Get the appropriate song names corresponding to predictions_label from the recommendation
# See recommendations_comb or recommendations_audio to get predictions_label
test_data['song_name'] = ""
for i in np.arange(0, len (test_data)):
  file_name = test_data.filename[i]
  song_names = file_name.replace(".mp3", "")
  test_data.song_name.loc[i] = song_names

# Get only songs found in predictions_label
genre_recom = test_data[test_data.song_name.isin(predictions_label)]

# Extract the genre tag for the songs
genre_recom = genre_recom[['song_name', 'tag']]

# Make recommendations from the same genre as the anchor song
for i in range(0,len(predictions_label)):
    anchor_name = genre_recom.song_name[i]
    anchor_genre = genre_recom.tag[i]
    one_genre = genre_recom[(genre_recom.tag == anchor_genre)]
    recommendation = sample(list(one_genre.song_name), 1)[0]
    genre_recommendations.loc[i] = [anchor_name, recommendation]
    # make sure the recommended song is not the anchor song; if so, sample again
    if anchor_name == recommendation:
        recommendation = sample(list(one_genre.song_name),1)[0]
    elif anchor_name != recommendation:
        genre_recommendations.loc[i] = [anchor_name, recommendation]
    else:
        continue

# Save genre recommendations
genre_recommendations.to_excel("/Users/helenk05/Desktop/Final_Folder/genre_recommendations.xlsx")