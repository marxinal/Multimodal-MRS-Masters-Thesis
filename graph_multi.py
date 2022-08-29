import os
import pandas as pd
import math
from PIL import Image
import numpy as np
import random
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from sklearn.metrics import confusion_matrix
from chart_studio import tools
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly
import plotly.offline
import plotly.io as pio
import re
import cv2
from sklearn.preprocessing import LabelEncoder
from numpy import load
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import TFDistilBertModel
from tqdm import tqdm

"""
Creating Plots for the Multimodal Model
"""

# Loading the model
model = load_model("/Users/helenk05/Desktop/Final_Folder/full_saved_model/Model_Full.h5", custom_objects={'TFDistilBertModel': TFDistilBertModel})
model.set_weights(model.get_weights())

# Get the test set of images created in Google Colab
test_x = load('/Users/helenk05/Desktop/Final_Folder/test.npz')
test_x = test_x['arr_0']

# Get the test set labels (genre tags) created in Google Colab
test_y = load('/Users/helenk05/Desktop/Final_Folder/test_test.npz')
test_y = test_y['arr_0']


# Make lyrics data
test_data = pd.read_excel("/Users/helenk05/Desktop/Final_Folder/test_data.xlsx")
train_data = pd.read_excel('/Users/helenk05/Desktop/Final_Folder/train_data.xlsx')
val_data = pd.read_excel('/Users/helenk05/Desktop/Final_Folder/val_data.xlsx')

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
train_data.lyrics = remove_brackets(train_data)
test_data.lyrics = remove_brackets(test_data)
val_data.lyrics = remove_brackets(val_data)

train_data.lyrics = get_space(train_data)
test_data.lyrics = get_space(test_data)
val_data.lyrics = get_space(val_data)

# Generate BERT input (ids, masks)
X_trainD_input_ids = np.zeros((len(train_data), 512))
X_trainD_attn_masks = np.zeros((len(train_data), 512))

X_testD_input_ids = np.zeros((len(test_data), 512))
X_testD_attn_masks = np.zeros((len(test_data), 512))

X_valD_input_ids = np.zeros((len(val_data), 512))
X_valD_attn_masks = np.zeros((len(val_data), 512))


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

X_trainD_input_ids,X_trainD_attn_masks = generate_dstil_data(train_data, X_trainD_input_ids, X_trainD_attn_masks, dbert_tokenizer)
X_testD_input_ids,X_testD_attn_masks = generate_dstil_data(test_data, X_testD_input_ids, X_testD_attn_masks, dbert_tokenizer)
X_valD_input_ids,X_valD_attn_masks = generate_dstil_data(val_data, X_valD_input_ids, X_valD_attn_masks, dbert_tokenizer)

# Make predictions based on the model
predictions = model.predict([test_x, X_testD_input_ids, X_testD_attn_masks], verbose=0)

# Get the relevant information from the model results
filename = "/Users/helenk05/Desktop/Final_Folder/full_saved_model/training_history_full.csv"
history = pd.read_csv(filename, header=0, low_memory=False)
history_array = history.values
epochs = history_array[:, 0]
training_accuracy = history_array[:, 2]
training_loss = history_array[:, 1]
val_accuracy = history_array[:, 4]
val_loss = history_array[:, 3]

# Sign in to generat4e images
py.sign_in('VikramShenoy','x1Un4yD3HDRT838vRkFA')

# Make the Accuracy Graph
trace0 = go.Scatter(
x = epochs,
y = training_accuracy,
mode = "lines",
name = "Training Accuracy",
marker = {'color': 'teal'}
)

trace1 = go.Scatter(
x = epochs,
y = val_accuracy,
mode = "lines",
name = "Validation Accuracy",
marker = {'color': 'paleturquoise'}
)

data = [trace0, trace1]
layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
fig = go.Figure(data=data, layout=layout)
fig['layout']['xaxis'].update(title="Number of Epochs", range = [0, 4], dtick=len(epochs)/5, showline = True, zeroline=True,  mirror='ticks', linecolor='#636363', linewidth=2)
fig['layout']['yaxis'].update(title="Accuracy", range = [0, 1], dtick=0.1, showline = True, zeroline=True, mirror='ticks',linecolor='#636363',gridcolor = '#D3D3D3', linewidth=2)
fig.update_layout(title = {
    'text': 'Accuracy for Multimodal Model',
    'y':0.85,
    'x':0.45,
    'xanchor': 'center',
    'yanchor': 'top'

})
# Saving the Accuracy graph
py.image.save_as(fig, filename="/Users/helenk05/Desktop/Final_Folder/full_saved_model/Graphs/Accuracy_Graph.png")
print ("Accuracy Graph Created")

# Make the Loss Graph
trace0 = go.Scatter(
x = epochs,
y = training_loss,
mode = "lines",
name = "Training Loss",
marker = {'color': 'purple'}
)

trace1 = go.Scatter(
x = epochs,
y = val_loss,
mode = "lines",
name = "Validation Loss",
marker = {'color': 'plum'}
)

data = [trace0, trace1]
layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
fig = go.Figure(data=data, layout=layout)
fig['layout']['xaxis'].update(title="Number of Epochs", range = [0, 4], dtick=len(epochs)/5, showline = True, zeroline=True,  mirror='ticks', linecolor='#636363', linewidth=2)
fig['layout']['yaxis'].update(title="Loss", dtick=0.1, showline = True, zeroline=True, mirror='ticks',linecolor='#636363',gridcolor = '#D3D3D3', linewidth=2)
fig.update_layout(title = {
    'text': 'Loss for Multimodal Model',
    'y':0.85,
    'x':0.45,
    'xanchor': 'center',
    'yanchor': 'top'

})
# Save the Loss Graph
py.image.save_as(fig, filename="/Users/helenk05/Desktop/Final_Folder/full_saved_model/Graphs/Loss_Graph.png")
print ("Loss Graph Created")

# Make the Confusion Matrix
y_pred = np.argmax(predictions, axis=1)
y_test = np.array([test_y[i] for i in range(0,len(test_y))])
y_pred = y_pred.tolist()
y_test = y_test.tolist()
cfm = confusion_matrix(y_test, y_pred)
labels = ["country", "pop", "rap", "rock"]

trace = go.Heatmap(z=cfm, x=labels, y=labels, reversescale=False, colorscale='Viridis')
data=[trace]
layout = go.Layout(
title = 'Confusion Matrix',
width = 800, height = 800,
showlegend = True,
xaxis = dict(dtick=1, tickangle=45),
yaxis = dict(dtick=1, tickangle=45))
fig = go.Figure(data=data, layout=layout)
fig.update_layout(title = {
    'text': 'Confusion Matrix for Multimodal Model',
    'y':0.9,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'

})
fig.update_xaxes(
    tickfont = dict(size=18)
)
fig.update_yaxes(
    tickfont = dict(size=18)
)
fig.update_layout(title_font_size=20)
py.image.save_as(fig, filename="/Users/helenk05/Desktop/Final_Folder/full_saved_model/Graphs/Confusion_Matrix.png")
print ("Confusion Matrix Created")
