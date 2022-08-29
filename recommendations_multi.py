import numpy as np
import keras
from keras.models import Model, load_model
import cv2
import re
import pandas as pd
from numpy import load
from numpy import mask_indices
import re
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import TFDistilBertModel
from keras import regularizers


# Load the trained model.
loaded_model = load_model("/Users/helenk05/Desktop/Final_Folder/full_saved_model/Model_Full.h5", custom_objects={'TFDistilBertModel': TFDistilBertModel})
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

# Load in Distill BERT tokenizer and model
dbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Get text (lyrics) data BERT input
# Remove info from brackets
def remove_brackets(data):
    for text in np.arange(0, len(data)):
        data.lyrics.loc[text] = re.sub("[\(\[].*?[\)\]]", "", data.lyrics[text])
    return data.lyrics

# Exchange /n for space
def get_space(data):
  for text in np.arange(0, len(data)):
    data.lyrics.loc[text] = data.lyrics.loc[text].replace("\n", " ")
  return data.lyrics

# Execute the functions
test_data.lyrics = remove_brackets(test_data)
test_data.lyrics = get_space(test_data)

# Generate BERT input (ids, masks)
X_testD_input_ids = np.zeros((len(test_data), 512))
X_testD_attn_masks = np.zeros((len(test_data), 512))

def generate_dstil_data(df, ids, masks, dbert_tokenizer):
    for i, text in tqdm(enumerate(df.lyrics)):
        tokenized_text = dbert_tokenizer.encode_plus(
            text,
            max_length=512, 
            truncation=True, 
            padding='max_length', 
            add_special_tokens=True,
            return_tensors='tf'
        )
        ids[i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask
    return ids, masks

X_testD_input_ids,X_testD_attn_masks = generate_dstil_data(test_data, X_testD_input_ids, X_testD_attn_masks, dbert_tokenizer)


# Assign images from test set, song names, BERT input to new variables
images, labels, input_ids, attn_masks = images_test, song_name, X_testD_input_ids, X_testD_attn_masks
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
        input_id = input_ids[i]
        attn_mask = attn_masks[i]
        test_image = np.expand_dims(test_image, axis=0)
        input_id = np.expand_dims(input_id, axis = 0)
        attn_mask = np.expand_dims(attn_mask, axis = 0)
         # Make a prediction using the data with the new model using all three inputs
        prediction = new_model.predict([test_image, input_id, attn_mask])
        predictions_song.append(prediction)
        counts.append(1)
    # Once the songs are stored in a new array, generate proper predictions 
    elif(labels[i] in predictions_label):
        index = predictions_label.index(labels[i])
        test_image = images[i]
        input_id = input_ids[i]
        attn_mask = attn_masks[i]
        test_image = np.expand_dims(test_image, axis=0)
        input_id = np.expand_dims(input_id, axis = 0)
        attn_mask = np.expand_dims(attn_mask, axis = 0)
        prediction = new_model.predict([test_image, input_id, attn_mask])
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
comb_recommendations = pd.DataFrame(columns = column_names)

# Create a for loop to generate recommendations 
for i in range(0,len(predictions_song)):
    prediction_anchor = predictions_song[i]
    prediction_anchor = prediction_anchor / counts[i]
    anchor_name = predictions_label[i]
    # Removing the selected (i) anchor 
    song_names = predictions_label.copy()
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
    comb_recommendations.loc[i] = [anchor_name, song_name]
    distance_array_2 = []
    print("Song name: " + str(song_name) + " with value " + str(value) + ' ' + str(index))

# Save the multimodal (combined) recommendations into an excel file
comb_recommendations.to_excel("/Users/helenk05/Desktop/Final_Folder/combined_recommendations.xlsx")