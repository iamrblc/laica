'''
This code processes your snippets:
- converts to mono
- normalizes the audio
- saves the processed audio to the processed folder
'''
# DEFINE WHERE YOUR FILES ARE
RAW_FOLDER = '/Users/rblc/code/iamrblc/laica/audio/snippet_test_set'

#################################
## DO NOT EDIT BELOW THIS LINE ##
#################################

# Import libraries
import os
from pydub import AudioSegment
from pydub.effects import normalize

# Define output directory
parent_dir = os.path.dirname(RAW_FOLDER)
processed_folder_name = 'snippets_processed'
PROCESSED_FOLDER = os.path.join(parent_dir, processed_folder_name)

def process_audio(input_folder = RAW_FOLDER, output_folder = PROCESSED_FOLDER):
    # create file list
    file_list = sorted(os.listdir(input_folder))
    print(f"Your folder has {len(file_list)} files, so this might take a while...")
    print("Grab a tea. I'll let you know whan it's done. :)")
    # loop through files in file list
    for file in file_list:
        if file.endswith('.wav'):
            # load the audio file with librosa
            audio = AudioSegment.from_file(os.path.join(input_folder, file),
                                        format="wav")

            # Convert to mono
            audio = audio.set_channels(1)

            # Normalize the audio using pydub's normalize effect
            normalized_audio = normalize(audio)

            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            # Save the normalized audio to a new file in the output folder
            output_path = os.path.join(output_folder, file)
            normalized_audio.export(output_path, format="wav")


process_audio()
print("Your audio files are processed. Have a nice day.")
