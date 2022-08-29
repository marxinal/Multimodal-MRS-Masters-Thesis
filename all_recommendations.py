import pandas as pd

# Load in recommendations
comb_recommendations = pd.read_excel('/Users/helenk05/Desktop/Final_Folder/combined_recommendations.xlsx')
audio_recommendations = pd.read_excel('/Users/helenk05/Desktop/Final_Folder/audio_recommendations.xlsx')
random_recommendations = pd.read_excel('/Users/helenk05/Desktop/Final_Folder/random_recommendations.xlsx')
genre_recommendations = pd.read_excel('/Users/helenk05/Desktop/Final_Folder/genre_recommendations.xlsx')

# Combine recommendations for the experiment 
all_recommendations = audio_recommendations.reset_index().merge(comb_recommendations, how = 'inner', 
                            left_on = 'anchor_song', right_on = 'anchor_song', 
                            suffixes = ('_audio', '_comb')).merge(random_recommendations,
                                                                    how = 'inner',
                                                                    left_on = 'anchor_song',
                                                                    right_on = 'anchor_song').merge(genre_recommendations,
                                                                    how = 'inner',
                                                                    left_on = 'anchor_song',
                                                                    right_on = 'anchor_song',
                                                                    suffixes = ('', '_genre'))


all_recommendations.drop(all_recommendations.filter(regex="Unname"), axis=1, inplace=True)

all_recommendations.to_excel('/Users/helenk05/Desktop/Final_Folder/all_recommendations.xlsx')

# Rename the files to get the pathway of 10-seconds audio
path = '/Users/helenk05/Desktop/Final_Folder/ten_second_songs/'

for i in range(0, len(all_recommendations)):
    all_recommendations.anchor_song.loc[i] = str(path + all_recommendations.anchor_song[i] + '_cut10.mp3')
    all_recommendations.recommendation_audio.loc[i] = str(path + all_recommendations.recommendation_audio[i] + '_cut10.mp3')
    all_recommendations.recommendation_comb.loc[i] = str(path + all_recommendations.recommendation_comb[i] + '_cut10.mp3')
    all_recommendations.recommendation.loc[i] = str(path + all_recommendations.recommendation[i] + '_cut10.mp3')
    all_recommendations.recommendation_genre.loc[i] = str(path + all_recommendations.recommendation_genre[i] + '_cut10.mp3')


all_recommendations.to_excel('/Users/helenk05/Desktop/Final_Folder/all_recommendations_exp.xlsx')