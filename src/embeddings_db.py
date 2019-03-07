""" 
This file contains the code for training the speaker verification lstm model

"""
import speaker_diarization
from configuration import get_config
import os
import numpy as np
import speaker_verification_lstm_model
import signal_processing
import librosa
import speaker_verfier

# get arguments from parser
config = get_config()   

def get_single_speaker_frames_from_utterances_path(utterances_folder):
    """ Extract frames of speach from folder with speaker uterrances
        Input: path to folder with a utternaces files
        Output: numpy array of frames
    """
    # for each one of the utterances extract spectrogram and get frames
    voice_utterances = []
    for single_utterance_file in os.listdir(utterances_folder):
        if os.path.isfile(os.path.join(utterances_folder, single_utterance_file)):
            utterance_path = os.path.join(utterances_folder, single_utterance_file)
            # extract audio 
            utterance_audio, _  = librosa.core.load(utterance_path, config.sr)
            voice_utterances.append(utterance_audio)

    #extract spectrogram
    spectrograms = signal_processing.extract_spectrograms_from_utterances(voice_utterances)
    speaker_frames = []
    # for each spectrogram save only first and last frames
    for single_spect in spectrograms:
        # take only utterrances that are long enough
        if single_spect.shape[0] > config.tisv_frame:
            speaker_frames.append(single_spect[:config.tisv_frame, :])    # first frames of partial utterance
            speaker_frames.append(single_spect[-config.tisv_frame:, :])   # last frames of partial utterance
    return np.concatenate( speaker_frames, axis=0)

def create_embeddings_db_from_calls(is_test=0):
    """ Use audio files to build DB of embeddings for each customer
    Input: Audio files path is saved in config
           is_test - only run the code, don't write file
    Output: Embeddings DB is saved in npy file in the path defined by config file 
    """
    # iterate over all existing calls and extract embeddings for each user
    embeddings = []
    # for each call create database of utterances
    for single_call_file in os.listdir(config.calls_path):
        call_file_path = os.path.join(config.calls_path, single_call_file)
        # Get embeddings of a single file, find mean embedding per speaker and append to the list
        single_speaker_embedding = speaker_verfier.get_single_speaker_embeddings_from_call_center_call(call_file_path)
        embeddings.append(single_speaker_embedding)
    #save the embedding into the DBs
    if is_test != 1:
        os.makedirs(config.embeddings_db_path, exist_ok=True)
        np.save(os.path.join(config.embeddings_db_path, "embedding_db"), embeddings)

def create_embeddings_db_from_utternaces(is_test=0):
    """ Use single speaker utterances files to build DB of embeddings for each speaker
    Input: Audio files path is saved in config and contains folder per speaker
           is_test - only run the code, don't write file
    Output: Embeddings DB is saved in npy file in the path defined by config file 
    """
    # iterate over all existing calls and extract embeddings for each speaker
    embeddings = []
    # Get embeddings of a single speaker, find mean embedding per speaker and append to the list
    for single_speaker_folder in os.listdir(config.utterance_path):
        single_speaker_folder = os.path.join(config.utterance_path, single_speaker_folder)
        single_speaker_frames = get_single_speaker_frames_from_utterances_path(single_speaker_folder)
        single_speaker_embeddings = speaker_verification_lstm_model.extract_embedding(single_speaker_frames)
        embeddings.append(np.mean(single_speaker_embeddings, axis=0))

    #save the embedding into the DBs
    if is_test != 1:
        print(" Created new embedding DB in ", config.embeddings_db_path, 
              ". There are", len(embeddings), "embeddings, with shape", embeddings[0].shape, "each.")
        os.makedirs(config.embeddings_db_path, exist_ok=True)
        np.save(os.path.join(config.embeddings_db_path, "embedding_db"), embeddings)

# Run to create utternaces database given a path to a folder with different speaker utterances or calls 
if __name__ == "__main__":
    if config.embeddings_from_calls:
        create_embeddings_db_from_calls(is_test=0)
    else:
        create_embeddings_db_from_utternaces(is_test=0)
