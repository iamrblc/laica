'''
This code takes a library of
- raw audio files in .wav format
- their corresponding textgrids
and turns them into snippets in a folder called snippets.
'''

'''
SECTION 1 - IMPORTS AND LOAD DATA

Define the folder to the raw files.
'''

# Path to folder containing audio files and textgrids
FOLDER = '/Users/rblc/code/iamrblc/laica/examples/splice_test/'


'''
DON'T CHANGE ANYTHING BELOW THIS LINE, UNLESS YOU KNOW WHAT YOU ARE DOING.
'''

###############################################################################


# Import all necessary libraries
from praatio import textgrid        # Handles textgrids
import pandas as pd                 # Handles dataframes
import numpy as np                  # Handles arrays
import os                           # Handles paths
from pydub import AudioSegment      # Creates snippets

# Create a list of file names called 'files'
def get_files(folder = FOLDER):
    files = sorted(os.listdir(folder))
    return files

files = get_files()

# Create a list of wav files and textgrid files
def wavs_and_textgrids(files = files):
    wavs = sorted([x for x in files if x.endswith(".wav")])
    textgrids = sorted([x for x in files if x.endswith(".TextGrid")])
    return wavs, textgrids

wavs, textgrids = wavs_and_textgrids()

# Check that the number of wav files and textgrid files match
assert len(wavs) == len(textgrids), "Number of wav files and textgrid files don't match. Check your folder and run again."

'''
SECTION 2 - CREATE A RAW DATAFRAME FROM TEXTGRIDS

DO NOT - I repeat - DO NOT change this section unless you are
fully aware of what you are doing. Which you are not.

The dataframe contains the following columns:
- file_name: the name of the textgrid file
- tier_name: the name of the tier in the textgrid
- start: the start time of the snippet
- end: the end time of the snippet
- label: the label of the snippet
'''

def textgrid_to_raw_df(textgrids = textgrids, folder = FOLDER):
    entry_dicts = []

    for i in range(len(textgrids)):

        textgrid_name = textgrids[i]
        file_path = os.path.join(folder, textgrid_name)
        try:
            tg = textgrid.openTextgrid(file_path, False)
            # loop through all possible tier levels
            for tier_name in tg.tierNames:
                tier = tg.getTier(tier_name)
                # append entry_dicts with a dictionary for each entry
                for entry in tier.entries:
                    entry_dict = {
                                'file_name': textgrid_name,
                                'tier_name': tier_name,
                                'start': entry.start,
                                'end': entry.end,
                                'label': entry.label}
                    entry_dicts.append(entry_dict)
        except:
            print(f'Error with {textgrid_name}')

    # create a pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(entry_dicts)

    # convert all labels to lowercase
    df = df.apply(lambda x: x.astype(str).str.lower())

    # convert start and end columns to float type
    df['start'] = df['start'].astype(float)
    df['end'] = df['end'].astype(float)

    return df

raw_df = textgrid_to_raw_df()

'''
SECTION 3 - FINETUNE THE RAW DATAFRAME FOR AUDIO SPLICING

The following section is based on careful EAD that is not included in this code.
It works only if you use the same original dataset as I did.
This section is absolutely necessary for reliable analysis.
If you don't have the same dataset, you will have to change this section.
'''

# Handle labels: unify namings and correct typos
raw_df['label'] = raw_df['label'].replace('whining', 'whine')
raw_df['label'] = raw_df['label'].replace('barking/yelping', 'yelp')
raw_df['label'] = raw_df['label'].replace('sound', 'bark')
raw_df['label'] = raw_df['label'].replace('growlgrowl', 'growl')
raw_df['label'] = raw_df['label'].replace('soundsound', 'bark')
raw_df['label'] = raw_df['label'].replace('soun', 'bark')
raw_df['label'] = raw_df['label'].replace('soundsoundsound', 'bark')
raw_df['label'] = raw_df['label'].replace('panting', 'pant')
raw_df['label'] = raw_df['label'].replace('sounds', 'bark')
raw_df['label'] = raw_df['label'].replace('other voclaization', 'other vocalization')
raw_df['label'] = raw_df['label'].replace('other vocalization', 'generic_dog_sound')

# Create a list of labels that are needed for future analysis
feature_wishlist = ['bark', 'whine', 'growl', 'pant', 'yelp']

# Create a splicer dataframe based on the feature wishlist
def splicer_df_maker(raw_df = raw_df, feature_wishlist = feature_wishlist):
    slicer_df = raw_df[raw_df['label'].isin(feature_wishlist)]
    return slicer_df

splicer_df = splicer_df_maker()

'''
SECTION 4 - CREATE SNIPPETS
'''

def export_audio_snippets(df = splicer_df, folder = FOLDER):
    # Create folder if it doesn't exist
    os.makedirs(os.path.join(folder, 'snippets'), exist_ok=True)

    # Group the dataframe by label
    label_groups = df.groupby('label')

    # Loop over label groups
    for label, group in label_groups:
        # Loop over rows in the group
        for i, row in group.iterrows():
            # Extract information from the row
            file_name = row['file_name']
            start = row['start']
            end = row['end']

            # Create wav file name from file_name
            wav_file_name = os.path.splitext(file_name)[0]+'.wav'

            # Create the path to the input audio file
            path = os.path.join(folder, wav_file_name)

            # Read the input audio file
            audio = AudioSegment.from_wav(path)

            # Compute the start and end times in milliseconds
            t1 = int(start * 1000)
            t2 = int(end * 1000)

            # Extract the audio snippet
            snippet = audio[t1:t2]

            # Create the output file name
            output_name = f"{label}_{i:05d}.wav"

            # Export the audio snippet to a file
            output_path = os.path.join(folder, 'snippets', output_name)
            snippet.export(output_path, format="wav")

export_audio_snippets()

'''
THE END OF TG TO SNIPPETS. THE BEGINNING OF SNIPPETS TO FEATURES.
'''
