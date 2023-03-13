'''
This code takes a library of snippets and extracts audio features from them.
If you don't have the snippets, run the code in 1_tg_to_snippet.py first.
'''

'''
SECTION 1 - IMPORTS AND LOAD DATA

Define the folder to the snippets.
'''

# Path to folder containing audio files and textgrids
folder = '/Users/rblc/code/iamrblc/laica/audio/snippets/'


'''
DON'T CHANGE ANYTHING BELOW THIS LINE, UNLESS YOU KNOW WHAT YOU ARE DOING.
'''

###############################################################################

# Import all necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

# Creating snippet dataframe
def snippet_df_maker(folder = folder):
    # Create empty lists to store the data
    file_names = []
    srs = []
    audios = []
    labels = []

    file_list = sorted(os.listdir(folder))

    for file in file_list:
        '''MAC creates a .DS_Store file in every folder, which causes an error
        this if statement skips the .DS_Store file'''
        if file.endswith('.wav'):
            # load the audio file with librosa
            audio, sr = librosa.load(os.path.join(folder, file))

            # split the filename into the label and ID columns
            label = file.split('_')[0]

            # append the extracted features to the lists
            file_names.append(file)
            srs.append(sr)
            audios.append(audio)
            labels.append(label)

            # convert the lists to a pandas dataframe
            df = pd.DataFrame(
                {'file_name':file_names,
                'sample_rate': srs,
                'audio': audios,
                'label': labels,})

            # add length column
            df['length'] = (df['audio'].apply(lambda x: len(x))/df['sample_rate'])

            # include only rows where length between 0.2 and 4 seconds
            df = df[df['length'] > 0.2]
            df = df[df['length'] < 4]

    return df

snippet_df = snippet_df_maker()

print(snippet_df.head())
