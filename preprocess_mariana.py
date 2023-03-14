import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gc

def preprocess(folder):
    # Create empty lists to store the data
    file_names = []
    audios = []
    labels = []

    file_list = sorted(file for file in os.listdir(folder) if file.endswith('.wav'))



    for file in file_list:

        # load the audio file with librosa
        audio, sr = librosa.load(os.path.join(folder, file))
        audio_norm = librosa.util.normalize(audio)
        # split the filename into the label and ID columns
        label = file.split('_')[0]#, file.split('_')[1].split('.')[0]


        file_names.append(file)
        audios.append(audio_norm)
        labels.append(label)


        # convert the lists to a pandas dataframe
        test_df = pd.DataFrame(
                {'file_name':file_names,
                'audio': audios,
                'label': labels})
        return test_df

def new_spect(audio_folder, spect_directory):

    df = preprocess(audio_folder)

    for i, row in df.iterrows():
        audio_file_name_without_extension = row["file_name"][:-4]

        y = row["audio"]

        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(spectrogram, y_axis='linear')

    plt.savefig(spect_directory + "/" + audio_file_name_without_extension + ".png")

    plt.clf()
    plt.close('all')
    gc.collect()
