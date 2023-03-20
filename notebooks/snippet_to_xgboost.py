##################
## TIMER STARTS ##
##################

import time
timer_start = time.perf_counter()

###############
## LIBRARIES ##
###############

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

import pickle

import librosa
import librosa.display
from librosa.effects import time_stretch, pitch_shift
import audiomentations as AA
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

import xgboost as xgb

# Setting up folder path
folder = '/Users/rblc/code/iamrblc/laica/audio/snippets_test_set'

############################
## BASE SNIPPET DATAFRAME ##
############################

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

            # normalize the audio with librosa normalizer
            audio = librosa.util.normalize(audio)

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

# Create df with only file_name, and label columns
snippet_df = snippet_df[['file_name', 'length', 'audio', 'label']]

# Create new df of audios between 0.2 and 4 seconds
snippet_df = snippet_df[(snippet_df['length'] >= 0.2) & (snippet_df['length'] <= 4)]

#####################
## AUGMENTING DATA ##
#####################

augmented_df = snippet_df.copy()

augmentation_pipeline = AA.Compose([
    AA.AddGaussianNoise(p=0.2, min_amplitude=0.001, max_amplitude=0.015),
    AA.TimeStretch(p = 0.2, min_rate=0.8, max_rate=1.2,),
    AA.PitchShift(p=0.3, min_semitones=-4, max_semitones=4),
    AA.Shift(p=0.3, min_fraction=-0.5, max_fraction=0.5)
])

default_augment = AA.AddGaussianNoise(p=1.0,
                                      min_amplitude=0.001,
                                      max_amplitude=0.015)

# Either pick the desired number of samples per class...
target_n = 100

# ...or adjust all to the class with the most samples.
#target_n = max(augmented_df.label.value_counts())


for label in augmented_df.label.unique():
    # create df from current label
    df_label = augmented_df[augmented_df['label'] == label].reset_index(drop=True)
    current_n = len(df_label)
    missing_n = target_n - current_n
    if missing_n > 0:
        for i in range(missing_n):
            # choose random row from df_label
            random_row = random.randint(0, current_n-1)
            # create augmented audio
            augmented_audio = augmentation_pipeline(df_label['audio'][random_row], sample_rate=22050)
            # check if any augmentations were applied
            if augmented_audio is None:
                # apply default augmentation
                augmented_audio = default_augment(df_label['audio'][random_row], sample_rate=22050)
            # create new row with augmented audio
            new_row = df_label.iloc[random_row].copy()
            new_row['audio'] = augmented_audio
            new_row['file_name'] = 'aug_' + new_row['file_name']
            # append new row to df
            augmented_df = augmented_df.append(new_row, ignore_index=True)

########################
## FEATURE EXTRACTION ##
########################

def extract_features(row):
    y, sr = row['audio'], 22050
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    tonal_centroid = librosa.feature.tonnetz(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    roll_off_frequency = librosa.feature.spectral_rolloff(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_vqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    rms_energy = librosa.feature.rms(y=y)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

    return mfcc, spectral_centroid, tonal_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness, roll_off_frequency, chroma_stft, chroma_cqt, chroma_cens, chroma_vqt, mel_spectrogram, rms_energy, tonnetz, zero_crossing_rate

# Apply the function to each row in the dataframe
features = augmented_df.apply(extract_features, axis=1)

# Add the features to the dataframe as new columns
augmented_df['mfcc'] = features.apply(lambda x: x[0])
augmented_df['spectral_centroid'] = features.apply(lambda x: x[1])
augmented_df['tonal_centroid'] = features.apply(lambda x: x[2])
augmented_df['spectral_bandwidth'] = features.apply(lambda x: x[3])
augmented_df['spectral_contrast'] = features.apply(lambda x: x[4])
augmented_df['spectral_flatness'] = features.apply(lambda x: x[5])
augmented_df['roll_off_frequency'] = features.apply(lambda x: x[6])
augmented_df['chroma_stft'] = features.apply(lambda x: x[7])
augmented_df['chroma_cqt'] = features.apply(lambda x: x[8])
augmented_df['chroma_cens'] = features.apply(lambda x: x[9])
augmented_df['chroma_vqt'] = features.apply(lambda x: x[10])
augmented_df['mel_spectrogram'] = features.apply(lambda x: x[11])
augmented_df['rms_energy'] = features.apply(lambda x: x[12])
augmented_df['tonnetz'] = features.apply(lambda x: x[13])
augmented_df['zero_crossing_rate'] = features.apply(lambda x: x[14])

'''# Saving all features to a pickle file if necessary

with open('all_features_snippets_raw.pkl', "wb") as file:
    pickle.dump(augmented_df, file)'''

#########################
## FEATURE ENGINEERING ##
#########################

engineered_df = augmented_df.copy()
engineered_df = engineered_df.drop(['audio', 'file_name', 'length'], axis=1)

def calculate_nested_stats(df, col_name):

    # Calculate median of first nested array only
    nested_median_func = lambda x: np.median(x[0])
    median_values = np.array(df[col_name].apply(nested_median_func).tolist())
    median_col_name = f"{col_name}_median"
    df[median_col_name] = pd.DataFrame(median_values)

    return df

for column_name in engineered_df.columns:
    if isinstance(engineered_df[column_name][0], np.ndarray):
        engineered_df = calculate_nested_stats(engineered_df, column_name)
        engineered_df = engineered_df.drop(columns = column_name)

engineered_df.head(2)


'''

THIS CODE CALCULATES THE MEDIAN OF EACH NESTED ARRAY INSTEAD OF JUST THE FIRST ONE
DON'T DELETE THIS. WE DON'T NEED THIS FOR THE PROTOTYPE BUT WE MIGHT NEED IT LATER


def calculate_nested_stats(df, col_name):

    # Calculate median values
    nested_median_func = lambda x: np.median(np.array(x), axis=1)
    median_values = np.array(df[col_name].apply(nested_median_func).tolist())
    num_cols = median_values.shape[1]
    median_col_names = [f"{col_name}_median_{i+1}" for i in range(num_cols)]
    df[median_col_names] = pd.DataFrame(median_values)

    return df

for column_name in engineered_df.columns:
    if isinstance(engineered_df[column_name][0], np.ndarray):
        engineered_df = calculate_nested_stats(engineered_df, column_name)
        engineered_df = engineered_df.drop(columns = column_name)'''


'''# Write engineered dataframe to pickle if necessary
with open('all_features_engineered_snippets_augmented.pkl', "wb") as file:
    pickle.dump(engineered_df, file)'''

####################
## MODEL PREPWORK ##
####################

df = engineered_df.copy()

# Encode the labels in a new column
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Somewhat complicated train/test split to prevent/minimize augmented data leakage

test_ratio = 0.2  # the percentage of data to be allocated to test set

groups = df.groupby('label_encoded')

train_indices = []
test_indices = []

# loop over each group and split it into train and test
for _, group in groups:
    n = len(group)
    n_test = int(np.ceil(n * test_ratio))  # number of samples to be allocated to test set
    n_train = n - n_test  # number of samples to be allocated to train set
    indices = group.index.to_list()
    test_indices += indices[:n_test]
    train_indices += indices[n_test:]

# sort the test indices of each group
test_indices = sorted(test_indices)

# create train and test dataframes
train_df = df.loc[train_indices]
test_df = df.loc[test_indices]

# Set X and y
X_train = train_df.drop(['label', 'label_encoded'], axis=1)
y_train = train_df['label_encoded']
X_test = test_df.drop(['label', 'label_encoded'], axis=1)
y_test = test_df['label_encoded']

######################
## SCALING THE DATA ##
######################

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

####################
## MODEL BUILDING ##
####################

model = xgb.XGBClassifier(booster = 'dart')
model.fit(X_train, y_train)

#####################################
## SAVING THE SCALER AND THE MODEL ##
#####################################

'''# Save the scaler
filename = '/Users/rblc/code/iamrblc/laica/scaler.pkl'
pickle.dump(scaler, open(filename, 'wb'))

# Save the model
filename = '/Users/rblc/code/iamrblc/laica/xgboost_model.pkl'
pickle.dump(model, open(filename, 'wb'))'''

################
## PREDICTION ##
################
encoded_classes = dict(zip(le.classes_, le.transform(le.classes_)))

y_proba = model.predict_proba(X_test)

# Define the encoded classes dictionary
encoded_classes = dict(zip(le.classes_, le.transform(le.classes_)))

# Invert the encoded classes dictionary to get a mapping of class indices to their labels
class_labels = {v: k for k, v in encoded_classes.items()}

# Print the predicted probabilities for each class with their corresponding labels and the actual label
for i, (proba, actual_label) in enumerate(zip(y_proba, y_test)):
    actual_label = le.inverse_transform([actual_label])[0]
    print(f"Sample {i+1} is a {actual_label}. The predicted probabilities:")
    for j, p in enumerate(proba):
        class_label = class_labels[j]
        print(f"{class_label}: {p*100:.4f}")

# Create a list of class labels for the legend
legend_labels = [class_labels[i] for i in range(len(class_labels))]

# Create a KDE plot for each class
for i in range(len(class_labels)):
    class_proba = y_proba[:, i]
    sns.kdeplot(class_proba, label=legend_labels[i])

# Set the plot title and axis labels
plt.title("KDE Plot of Predicted Probabilities")
plt.xlabel("Probability")
plt.ylabel("Density")

# Add a legend to the plot
plt.legend()

# Show the plot
plt.show()

################
## TIMER ENDS ##
################

timer_ends = time.perf_counter()

duration = timer_ends - timer_start

hours, remainder = divmod(duration, 3600)
minutes, seconds = divmod(remainder, 60)

print(f'Your program ran for {duration: .2f} seconds which is {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.')
