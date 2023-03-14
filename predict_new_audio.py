
import pandas as pd
import numpy as np

import pickle

import librosa
import librosa.display

# Load the trained model
filename = '/Users/rblc/code/iamrblc/laica/xgboost_model.pkl'
model = pickle.load(open(filename, 'rb'))
scaler = pickle.load(open('/Users/rblc/code/iamrblc/laica/scaler.pkl', 'rb'))

# Provide path to new audio file
new_audio = '/Users/rblc/code/iamrblc/laica/audio/snippets_from_yt/deep_growl.wav'
actual_label = 'growl'

# Load the audio file
audio, sr = librosa.load(new_audio)

# Normalize the audio with librosa
audio = librosa.util.normalize(audio)

# Make dataframe
df = pd.DataFrame({'audio': [audio]})

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
features = df.apply(extract_features, axis=1)

# Add the features to the dataframe as new columns
df['mfcc'] = features.apply(lambda x: x[0])
df['spectral_centroid'] = features.apply(lambda x: x[1])
df['tonal_centroid'] = features.apply(lambda x: x[2])
df['spectral_bandwidth'] = features.apply(lambda x: x[3])
df['spectral_contrast'] = features.apply(lambda x: x[4])
df['spectral_flatness'] = features.apply(lambda x: x[5])
df['roll_off_frequency'] = features.apply(lambda x: x[6])
df['chroma_stft'] = features.apply(lambda x: x[7])
df['chroma_cqt'] = features.apply(lambda x: x[8])
df['chroma_cens'] = features.apply(lambda x: x[9])
df['chroma_vqt'] = features.apply(lambda x: x[10])
df['mel_spectrogram'] = features.apply(lambda x: x[11])
df['rms_energy'] = features.apply(lambda x: x[12])
df['tonnetz'] = features.apply(lambda x: x[13])
df['zero_crossing_rate'] = features.apply(lambda x: x[14])

df = df.drop(['audio'], axis=1)

def calculate_nested_stats(df, col_name):

    # Calculate median of first nested array only
    nested_median_func = lambda x: np.median(x[0])
    median_values = np.array(df[col_name].apply(nested_median_func).tolist())
    median_col_name = f"{col_name}_median"
    df[median_col_name] = pd.DataFrame(median_values)

    return df

for column_name in df.columns:
    if isinstance(df[column_name][0], np.ndarray):
        df = calculate_nested_stats(df, column_name)
        df = df.drop(columns = column_name)

# Add scaler
df = pd.DataFrame(scaler.transform(df), columns=df.columns)
encoded_classes = {'bark': 0, 'growl': 1, 'pant': 2, 'whine': 3, 'yelp': 4}

# Run the model on the new audio file
prediction = model.predict(df)

# Calculate the probabilities of each class
proba = model.predict_proba(df)

# Invert the encoded classes dictionary to get a mapping of class indices to their labels
class_labels = {v: k for k, v in encoded_classes.items()}

print(f"The provided snippet is a {actual_label}. The predicted probabilities:")
for i, p in enumerate(proba[0]):
    class_label = class_labels[i]
    print(f"{class_label}: {p:.6f}")
