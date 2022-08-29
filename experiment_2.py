from playsound import playsound
import multiprocessing
import random
import pandas as pd

all_recommendations = pd.read_excel('/Users/helenk05/Desktop/Final_Folder/all_recommendations_exp.xlsx')

anchor_q = []
audio_q = []
comb_q = []
genre_q = []

for i in range(599, 1199):
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
    if (anchor == 1 and i <= 1199):
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
        # genre condition
        def genre_fun():
            genre_rec = all_recommendations.recommendation_genre[i]
            p_random = multiprocessing.Process(target = playsound, args = (genre_rec,))
            p_random.start()
            try:
                genre_rec = input("Do you like this song?")
                genre_rec = int(genre_rec)
                genre_q.append(genre_rec)
                p_random.terminate()
            except:
                print("slip")
                p_random.terminate()
                genre_q.append("slip")
        # create a function to randomize the conditions
        conditions = [audio_fun, comb_fun, genre_fun]
        def shuffle(conditions):
            random.shuffle(conditions)
            for i in conditions:
                i()
        shuffle(conditions = conditions)
    # when the participant doesn't like the anchor song, don't play the recommendations
    elif i <= 1199:
        audio_q.append("no")
        comb_q.append("no")
        genre_q.append("no")
        continue
    # once 600 trials are reached exit the loop
    else:
        print("Thank you for participating!")
        anchor_q = anchor_q[:-1]
        break

participant2 = pd.DataFrame({'anchor': list(anchor_q), 'audio': list(audio_q), 'comb': list(comb_q), 'random': list(genre_q)}, columns=['anchor', 'audio', 'comb', 'random'])

participant2.to_excel('/Users/helenk05/Desktop/Final_Folder/participant2_exp.xlsx')









