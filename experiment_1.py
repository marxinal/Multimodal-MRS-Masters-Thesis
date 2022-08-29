from playsound import playsound
import multiprocessing
import random
import pandas as pd

"""
This is a coding setup for the computer-based experiment. Participants were played an anchor song, 
and three other songs from the MRSs. The participants were instructed to respond with 1 and 0s. 
"""


all_recommendations = pd.read_excel('/Users/helenk05/Desktop/Final_Folder/all_recommendations_exp.xlsx')

anchor_q = []
audio_q = []
comb_q = []
random_q = []

for i in range(0, 599):
    # select an anchor song
    anchor_song = all_recommendations.anchor_song[i]
    # play the anchor song
    p = multiprocessing.Process(target = playsound, args = (anchor_song,))
    p.start()
    try:
        # ask the participant whether they like the song 
        anchor = input("Do you like this song?")
        # get the 0 or 1 as an answer
        anchor = int(anchor)
        # keep track of the answer
        anchor_q.append(anchor)
        p.terminate()
    # if the participant accidentally presses something else but a number continue
    except:
        print("slip")
        p.terminate()
        anchor_q.append("slip")
    # when the participant likes the anchor song play the recommended songs
    if (anchor == 1 and i <= 600):
        # audio-only condition
        def audio_fun():
            audio = all_recommendations.recommendation_audio[i]
            p_audio = multiprocessing.Process(target = playsound, args = (audio,))
            p_audio.start()
            try:
                audio = input("Do you like this song?")
                audio = int(audio)
                audio_q.append(audio)
                p_audio.terminate()
            except:
                print("slip")
                p_audio.terminate()
                audio_q.append("slip")
        # multimodal (combined) condition
        def comb_fun():
            comb = all_recommendations.recommendation_comb[i]
            p_comb = multiprocessing.Process(target = playsound, args = (comb,))
            p_comb.start()
            try:
                comb = input("Do you like this song?")
                comb = int(comb)
                comb_q.append(comb)
                p_comb.terminate()
            except:
                print("slip")
                p_comb.terminate()
                comb_q.append("slip")
        # random condition
        def random_fun():
            random_rec = all_recommendations.recommendation[i]
            p_random = multiprocessing.Process(target = playsound, args = (random_rec,))
            p_random.start()
            try:
                random_rec = input("Do you like this song?")
                random_rec = int(random_rec)
                random_q.append(random_rec)
                p_random.terminate()
            except:
                print("slip")
                p_random.terminate()
                random_q.append("slip")
        # create a function to randomize the conditions
        conditions = [audio_fun, comb_fun, random_fun]
        def shuffle(conditions):
            random.shuffle(conditions)
            for i in conditions:
                i()
        shuffle(conditions = conditions)
    # when the participant doesn't like the anchor song, don't play the recommendations
    elif i <= 600:
        audio_q.append("no")
        comb_q.append("no")
        random_q.append("no")
        continue
    # once 600 trials are reached exit the loop
    else:
        print("Thank you for participating!")
        anchor_q = anchor_q[:-1]
        break

participant1 = pd.DataFrame({'anchor': list(anchor_q), 'audio': list(audio_q), 'comb': list(comb_q), 'random': list(random_q)}, columns=['anchor', 'audio', 'comb', 'random'])

participant1.to_excel('/Users/helenk05/Desktop/Final_Folder/participant1_exp.xlsx')



