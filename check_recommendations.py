
# Checking how many times the recommendations were the same (multi vs audio)
import numpy as np
import pandas as pd 

# Load in all_recommendations
all_recommendations = pd.read_excel("/Users/helenk05/Desktop/Final_Folder/all_recommendations.xlsx")

all_reco_p1 = all_recommendations[0:599]
comparison_column_p1 = np.where(all_reco_p1["recommendation_audio"] == all_reco_p1["recommendation_comb"], True, False)
sum(comparison_column_p1)

all_reco_p2 = all_recommendations[599:1199]
comparison_column_p2 = np.where(all_reco_p2["recommendation_audio"] == all_reco_p2["recommendation_comb"], True, False)
sum(comparison_column_p2)