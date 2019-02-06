import argparse
import numpy as np
import configparser

parser = argparse.ArgumentParser()    # make parser

# get arguments
def get_config():
    config, unparsed = parser.parse_known_args()
    return config

# return bool type of argument
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


config_defaults = configparser.ConfigParser()
config_defaults.read('../configs/config.cnf')
config_defaults.sections()


# Signal Processing
signal_processing_arg = parser.add_argument_group('Signal Processing')
signal_processing_arg.add_argument('--sr', type=int, default=config_defaults['Signal Processing']['sr'], help="sampling rate")
signal_processing_arg.add_argument('--nfft', type=int, default=config_defaults['Signal Processing']['nfft'], help="fft kernel size")
signal_processing_arg.add_argument('--window', type=float, default=config_defaults['Signal Processing']['window'], help="window length (ms)")
signal_processing_arg.add_argument('--hop', type=float, default=config_defaults['Signal Processing']['hop'], help="hop size (ms)")
signal_processing_arg.add_argument('--tisv_frame', type=int, default=config_defaults['Signal Processing']['tisv_frame'], help="max frame number of utterances of tisv")
signal_processing_arg.add_argument('--dialing_tone_thresh', type=int, default=config_defaults['Signal Processing']['dialing_tone_thresh'], help="Threshold for ringtone identification in a call")
signal_processing_arg.add_argument('--mel_nof', type=int, default=config_defaults['Signal Processing']['mel_nof'], help="Number of Mel-frequency bins")

# Model Parameters
model_arg = parser.add_argument_group('Model')
model_arg.add_argument('--hidden', type=int, default=config_defaults['Model Parameters']['hidden'], help="hidden state dimension of lstm")
model_arg.add_argument('--proj', type=int, default=config_defaults['Model Parameters']['proj'], help="projection dimension of lstm")
model_arg.add_argument('--num_layer', type=int, default=config_defaults['Model Parameters']['num_layer'], help="number of lstm layers")
model_arg.add_argument('--model_path', type=str, default=config_defaults['Model Parameters']['model_path'], help="model directory to save or load")
model_arg.add_argument('--model_num', type=int, default=config_defaults['Model Parameters']['model_num'], help="number of ckpt file to load")

# Testing Parameters
test_arg = parser.add_argument_group('Testing')
test_arg.add_argument('--diarization_test_path', type=str, default=config_defaults['Testing Parameters']['diarization_test_path'], help="Path to file for diarization test")
test_arg.add_argument('--calls_to_compare_path', type=str, default=config_defaults['Testing Parameters']['calls_to_compare_path'], help="Folder containing two audio files of calls that should be compared")

# Creating Embeddings Database
embedding_db_arg = parser.add_argument_group('Embedding_DB')
embedding_db_arg.add_argument('--embeddings_db_path', type=str, default=config_defaults['Embeddings Database']['embeddings_db_path'], help="path of embedding DB")
embedding_db_arg.add_argument('--embeddings_from_calls', type=str, default=config_defaults['Embeddings Database']['embeddings_from_calls'], help="if true create embedding db from calls. if false create embedding db from utterances")
embedding_db_arg.add_argument('--utterance_path', type=str, default=config_defaults['Embeddings Database']['utterance_path'], help="path to folder with different speakers utterances to create embedding DB from")
embedding_db_arg.add_argument('--calls_path', type=str, default=config_defaults['Embeddings Database']['calls_path'], help="path to folder with calls of different speakers to create embedding DB from")

config = get_config()
print(config)           # print all the arguments
