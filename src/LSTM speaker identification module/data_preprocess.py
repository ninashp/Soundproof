import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from configuration import get_config
from utils import keyword_spot

config = get_config()   # get arguments from parser

# downloaded dataset path
audio_path= '/home/nina/Dev/insight/speech_data/RedCarpet/audio_test'              # utterance dataset
sr = 44100 #sampling rate

test_path = '/home/nina/Dev/insight/speech_data/RedCarpet/python_spec' 

def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """
    print("start text independent utterance feature extraction")
    os.makedirs(test_path, exist_ok=True)   # make folder to save train file

    utter_min_len = (config.tisv_frame * config.hop + config.window) * config.sr    # lower bound of utterance length
    total_speaker_num = len(os.listdir(audio_path))
    print("total speaker number : %d"%total_speaker_num)
    for i, folder in enumerate(os.listdir(audio_path)):
        speaker_path = os.path.join(audio_path, folder)     # path of each speaker
        print("%dth speaker processing..."%i)
        utterances_spec = []
        for utter_name in os.listdir(speaker_path):
            utter_path = os.path.join(speaker_path, utter_name)         # path of each utterance
            utter_, _sr = librosa.core.load(utter_path, sr)        # load utterance audio
            utter = librosa.resample(utter_, sr, config.sr)
            intervals = librosa.effects.split(utter, top_db=20)         # voice activity detection
            for interval in intervals:
                if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                    utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                    S = librosa.core.stft(y=utter_part, n_fft=config.nfft,
                                          win_length=int(config.window * config.sr), hop_length=int(config.hop * config.sr))
                    S = np.abs(S) ** 2
                    mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
                    S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances

                    utterances_spec.append(S[:, :config.tisv_frame])    # first 180 frames of partial utterance
                    utterances_spec.append(S[:, -config.tisv_frame:])   # last 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        np.save(os.path.join(test_path, "speaker%d.npy"%(i)), utterances_spec)


if __name__ == "__main__":
    save_spectrogram_tisv()