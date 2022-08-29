import numpy as np
import keras
from keras.models import Model, load_model
import cv2
import re
import pandas as pd
from numpy import load

# Load the trained model.
loaded_model = load_model("/Users/helenk05/Desktop/Final_Folder/audio_saved_model/Model_Audio.h5")
loaded_model.set_weights(loaded_model.get_weights())
# Discard the Softmax layer, Second last layer provides the latent feature
# representation.
matrix_size = loaded_model.layers[-2].output.shape[1]
new_model = Model(loaded_model.inputs, loaded_model.layers[-2].output)
print (new_model.summary())

# Load test data
test_data = pd.read_excel("/Users/helenk05/Desktop/Final_Folder/test_data.xlsx")

# Get the test set of images created in Google Colab
test_x = load('/Users/helenk05/Desktop/Final_Folder/test_images.npz')
test_x = test_x['arr_0']
images_test = np.array(test_x)

# Load the names of the songs created in Google Colab
song_name = load('/Users/helenk05/Desktop/Final_Folder/song_names.npz')
song_name = song_name['arr_0']

# Assign images from test set and song names to new variables
images, labels = images_test, song_name
images = np.expand_dims(images, axis=3)
# Normalize the image.
images = images / 255.
# Display list of available test songs.
print (np.unique(labels))
# Create an empty matrix to store the anchor
prediction_anchor = np.zeros((1, matrix_size))
count = 0
predictions_song = []
predictions_label = []
counts = []
distance_array = []

for i in range(0, len(labels)):
    if(labels[i] not in predictions_label):
        predictions_label.append(labels[i])
        test_image = images[i]
        test_image = np.expand_dims(test_image, axis=0)
        # Make a prediction using the data with the new model
        prediction = new_model.predict(test_image)
        predictions_song.append(prediction)
        counts.append(1)
    # Once the songs are stored in a new array, generate proper predictions 
    elif(labels[i] in predictions_label):
        index = predictions_label.index(labels[i])
        test_image = images[i]
        test_image = np.expand_dims(test_image, axis=0)
        prediction = new_model.predict(test_image)
        predictions_song[index] = predictions_song[index] + prediction
        counts[index] = counts[index] + 1



# Create necessary empty lists for the for loop
prediction_songs_2 = []
distance_array_2 = []
predictions = []
counts_2 = []
song_names = []

# Create a dataframe to store the recommended songs after the for loop
column_names = ['anchor_song', 'recommendation']
audio_recommendations = pd.DataFrame(columns = column_names)

# Create a for loop to generate recommendations 
for i in range(0,len(predictions_song)):
    prediction_anchor = predictions_song[i]
    prediction_anchor = prediction_anchor / counts[i]
    anchor_name = predictions_label[i]
    song_names = predictions_label.copy()
    # Removing the selected anchor
    song_names.pop(i)
    # Save songs (without anchor) to a new list, to generate recommenations
    prediction_songs_2 = predictions_song.copy()
    prediction_songs_2.pop(i)
    # Get rid of anchor count 
    counts_2 = counts.copy()
    counts_2.pop(i)
    # Loop over songs (without anchor)
    for j in range(0,len(prediction_songs_2)):
        print(i, " ", j)
        # Averaging the latent feature representations of the songs
        prediction_songs_2[j] = prediction_songs_2[j] / counts_2[j]
        # Find cosine similarity between anchor and the rest of the songs
        distance_array_2.append(np.sum(prediction_anchor * prediction_songs_2[j]) / (np.sqrt(np.sum(prediction_anchor**2)) * np.sqrt(np.sum(prediction_songs_2[j]**2))))
    distance_array_2 = np.array(distance_array_2)
    # Getting the song with the highest cosine similarity with the anchor
    index = np.argmax(distance_array_2)
    value = distance_array_2[index]
    song_name = song_names[index]
    # Saving the anchor song and the recommended song 
    audio_recommendations.loc[i] = [anchor_name, song_name]
    distance_array_2 = []
    print("Song name: " + str(song_name) + " with value " + str(value) + ' ' + str(index))

# Save the audio recommendations into an excel file
audio_recommendations.to_excel("/Users/helenk05/Desktop/Final_Folder/audio_recommendations.xlsx")