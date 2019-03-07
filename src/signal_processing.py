""" 
This file contains the code for signal processing module of application.

"""
import numpy as np
from configuration import get_config
import scipy
import librosa

config = get_config()   # get arguments from parser

# constants for output of identify_call_type
FIRST_SPEAKER_CUSTOMER = 0
FIRST_SPEAKER_REPRESENTATIVE = 1

def print_call_type(call_type):
    """ Used to print call type
        Input: call type int
    """
    if call_type == FIRST_SPEAKER_CUSTOMER:
        print("First speaker is customer")
    else:
        print("First speaker is representative")

def identify_call_type(call_file):
    """ Identify who speaks first in the call according to phone tone.
        If a call starts with a dial tone customer speaks first, else representative.
        Input: path to a call
        Output: FIRST_SPEAKER_CUSTOMER if customer speaks first,
                FIRST_SPEAKER_REPRESENTATIVE if representetor speaks first
    """
    # from the first frame of sound measure 1.5 sec and look for 400Hz tone
    nof_frames = librosa.core.time_to_samples(1.5, sr=config.sr)

    call_audio, _  = librosa.core.load(call_file, config.sr)
    intervals = librosa.effects.split(call_audio, top_db = 20)

    tone_fft = scipy.fft(call_audio[intervals[0][0]:intervals[0][0]+nof_frames])
    tone_fft_mag = np.absolute(tone_fft)   # spectral magnitude
    f = np.linspace(0, config.sr, nof_frames)  # frequency variable
    if (round(f[np.argmax(tone_fft_mag)]) == 400 and max(tone_fft_mag)>config.dialing_tone_thresh):
        # dialing tone detected! this means represntative is calling to the customer, customer speaks first
        return FIRST_SPEAKER_CUSTOMER
    else:
        # this means customer is calling to the call center, represntative speaks first
        return FIRST_SPEAKER_REPRESENTATIVE

def extract_utterances_from_a_call(call_file):
    """ Get a file, output a numpy array with frames exreacted from the call of voice.
        The frames are utterances of minimal length, split by 20DB limit
        Input: audio file path 
        Output: list of numpy arrays, each of them representing a single speech utterance
                list of numpy array representing the timestamp start and end of each utterance in the call
    """
    # extract audio 
    call_audio, _  = librosa.core.load(call_file, config.sr)
    # split the audio to voice and no-voice according to amplitude
    intervals = librosa.effects.split(call_audio, top_db = 20)
    # lower bound of utterance length - below that discard
    utter_min_len = (config.tisv_frame * config.hop + config.window) * config.sr   

    utterances_list = []
    utterances_timestamps = []
    for interval in intervals:
        # Check that utterance length is sufficient 
        if (interval[1]-interval[0]) > utter_min_len:
            utterances_list.append(call_audio[interval[0]:interval[1]])
            utterances_timestamps.append(np.array([librosa.core.samples_to_time(interval[0], sr=config.sr),
                                                    librosa.core.samples_to_time(interval[1], sr=config.sr)]))
    return utterances_list, utterances_timestamps

def extract_spectrograms_from_utterances(utterances_list):
    """ Get a list of utterances and extract spectrograms binned in mel-binning for each frame 
        Input: list of numpy arrays, each of them representing a single speech utterance  
        Output: list of numpy arrays, each of them representing a spectrogram of a single speech utterance 
    """
    spectrograms_list = []
    # iterate on all utterances, extract spectrogram from each
    for utterance in utterances_list:
        spect = librosa.core.stft(y=utterance, n_fft = config.nfft, 
                                    win_length=int(config.window * config.sr), hop_length=int(config.hop * config.sr))
        spect = np.abs(spect) ** 2
        mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=config.mel_nof)
        # log mel spectrogram of utterances
        spect_bins = np.log10(np.dot(mel_basis, spect) + 1e-6)           
        spectrograms_list.append(np.transpose(spect_bins))
    return spectrograms_list

def split_segment_to_frames(seg):
    """ Given an audio segment, split it into frames according to size config.tisv_frame
        Input: seg - audio segment
        Output: list of frames
    """
    # Extrct spectrogram
    spect = np.transpose(extract_spectrograms_from_utterances([seg])[0])
    #Get config.tisv_frame STFT windows with 50% overlap
    STFT_frames = []

    for j in range(0, spect.shape[1], int(.12/config.hop)):
        if j + config.tisv_frame < spect.shape[1]:
            STFT_frames.append(np.transpose(spect[:,j:j+config.tisv_frame]))
        else:
            break
    return STFT_frames
    