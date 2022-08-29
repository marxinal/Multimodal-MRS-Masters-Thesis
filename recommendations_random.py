from random import sample
import pandas as pd

# Create empty data frame to store random recommendations
column_names = ['anchor_song', 'recommendation']
random_recommendations = pd.DataFrame(columns = column_names)

# Generate random recommendations
for i in range(0,len(predictions_label)):
    anchor_name = predictions_label[i]
    song_name = sample(predictions_label, 1)[0]
    song_name = song_name.strip("['']")
    random_recommendations.loc[i] = [anchor_name, song_name]
    # Make sure the recommendation is not the same as the anchor song
    if anchor_name == song_name:
        song_name = sample(predictions_label,1)[0]
        song_name = song_name.strip("['']")
    elif anchor_name != song_name:
        random_recommendations.loc[i] = [anchor_name, song_name]
    else:
        continue

# Save random recommendations
random_recommendations.to_excel("/Users/helenk05/Desktop/Final_Folder/random_recommendations.xlsx")
    