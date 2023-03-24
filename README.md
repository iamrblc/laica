# LAICA - Learning-based Artificial Intelligence on Canine Acoustics

## The problem
Researchers studying canine vocalizations need to listen to lenghty audio files and manually label dog sounds of their interests in Praat.

## Our solution
We created two machine learning models (an XGBoost Classifier and a Convolutional Neural Network) that can classify dog sound snippets (barks, growls,
whines and pants) regardless of dog breeds.

We also created a protype that can label lenghty audio files and return both a Pandas dataframe and a Praat compatible TextGrid file for future use.

## The data
We received 26 GB of audio files with corresponding TextGrid label files from the Ethology Department of Budapest. 

## The challenges
- pairing audio files with corresponding label files (various naming conventions, inconvenient directory structure)
- converting the TextGrid file labels into a usable format
- splicing out dog sound snippets (0.2 - 4 second sounds of interest)
- cleaning, cleaning, cleaning and audio processing
- sample imbalance correction 
- exploring the most effective way of find sounds of interest in a long audio file 
- creating the label output file that's compatible with Praat

## Limitations
Although our __models__ are fairly limited at the moment and they can only classify dog sounds accurately (all other sounds will be squeezed into either of 
these categories), we created a solid framework for future development.
Our __labelling script__ is only a prototype that works on specifically selected features at the moment. It was not meant to be a working product,
rather a tech showcase to the clients to serve as a base for future discussions and clarifying exact requirements.

## Play with LAICA
Feel free to test it at [https://laica-pred.streamlit.app/](https://laica-pred.streamlit.app/). You can upload a few second long dog sound sample in 
wav format and choose which model you'd like to test (Audio features for XGBoost, Spectrogram for the CNN).
