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

"""
Creating Plots for the Audio-only Model
"""

# Loading the model
model = load_model("/Users/helenk05/Desktop/Final_Folder/audio_saved_model/Model_Audio.h5")
model.set_weights(model.get_weights())

# Get the test set of images created in Google Colab
test_x = load('/Users/helenk05/Desktop/Final_Folder/test.npz')
test_x = test_x['arr_0']

# Get the test set labels (genre tags) created in Google Colab
test_y = load('/Users/helenk05/Desktop/Final_Folder/test_test.npz')
test_y = test_y['arr_0']

# Make model predictions on the test set
predictions = model.predict(test_x, verbose=0)

# Get the relevant information from the model results
filename = "/Users/helenk05/Desktop/Final_Folder/audio_saved_model/training_history_audio.csv"
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
fig['layout']['xaxis'].update(title="Number of Epochs", range = [0, 14], dtick=len(epochs)/15, showline = True, zeroline=True,  mirror='ticks', linecolor='#636363',linewidth=2)
fig['layout']['yaxis'].update(title="Accuracy", range = [0, 1], dtick=0.1, showline = True, zeroline=True, mirror='ticks',linecolor='#636363',  gridcolor = '#D3D3D3', linewidth=2)
fig.update_layout(title = {
    'text': 'Accuracy for Audio-Only Model',
    'y':0.85,
    'x':0.45,
    'xanchor': 'center',
    'yanchor': 'top'

})
# Saving the Accuracy graph
py.image.save_as(fig, filename="/Users/helenk05/Desktop/Final_Folder/audio_saved_model/Graphs/Accuracy_Graph.png")
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
fig['layout']['xaxis'].update(title="Number of Epochs", range = [0, 14], dtick=len(epochs)/15, showline = True, zeroline=True,  mirror='ticks', linecolor='#636363', linewidth=2)
fig['layout']['yaxis'].update(title="Loss", dtick=0.1, showline = True, zeroline=True, mirror='ticks',linecolor='#636363',gridcolor = '#D3D3D3',linewidth=2)
fig.update_layout(title = {
    'text': 'Loss for Audio-Only Model',
    'y':0.85,
    'x':0.45,
    'xanchor': 'center',
    'yanchor': 'top'

})
# Saving the Loss Graph
py.image.save_as(fig, filename="/Users/helenk05/Desktop/Final_Folder/audio_saved_model/Graphs/Loss_Graph.png")
print ("Loss Graph Created")


# Make the Confusion Matrix 
y_pred = np.argmax(predictions, axis=1)
y_test = np.array([test_y[i] for i in range(0,len(test_y))])
y_pred = y_pred.tolist()
y_test = y_test.tolist()
cfm = confusion_matrix(y_test, y_pred)
labels = ["country", "pop", "rap", "rock"]

trace = go.Heatmap(z=cfm, x=labels, y=labels, reversescale=False, colorscale="Viridis")
data=[trace]
layout = go.Layout(
title = 'Confusion Matrix',
width = 800, height = 800,
showlegend = True,
xaxis = dict(dtick=1, tickangle=45),
yaxis = dict(dtick=1, tickangle=45))
fig = go.Figure(data=data, layout=layout)
fig.update_layout(title = {
    'text': 'Confusion Matrix for Audio-Only Model',
    'y':0.9,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top',

})
fig.update_xaxes(
    tickfont = dict(size=18)
)
fig.update_yaxes(
    tickfont = dict(size=18)
)
fig.update_layout(title_font_size=20)
# Saving the Confusion Matrix
py.image.save_as(fig, filename="/Users/helenk05/Desktop/Final_Folder/audio_saved_model/Graphs/Confusion_Matrix.png")
print ("Confusion Matrix Created")