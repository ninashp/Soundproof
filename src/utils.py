""" 
This file contains miscellaneous utilities.

"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import librosa
from librosa import display
from configuration import get_config
import pandas as pd
import re

# get arguments from parser
config = get_config()  

def plot_spectrogram(audio):
    """ Plot spectrogram for audio input
        Input: audio file
        Output: None
    """
    plt.figure(figsize=(12, 4))
    S_full, _ = librosa.magphase(librosa.stft(audio))
    librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                            y_axis='log', x_axis='time', sr=config.sr)
       
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def get_pandas_ds_from_trn(trn_file):
    """ Arrange .trn file in pandas dataframe
        Input: Path to .trn file
        Output: pandas dataframe with start, end times and speaker id for each time section
    """
    data = pd.read_csv(trn_file, sep="\t", header = None)

    # Remove text
    data = data.iloc[:,0:len(data.columns)-1]
    data = data.fillna("")

    # find first speaker
    first_speaker = data.iloc[0,len(data.columns)-1]

    diar_ref = data
    # Fill all lines with speaker
    for line in range(data.shape[0]):
        if re.search('[a-zA-Z]', data.iloc[line,len(data.columns)-1]):
            # If speaker is not empty, save him
            if data.iloc[line,len(data.columns)-1] == first_speaker:
                current_speaker = 0
            else:
                current_speaker = 1
            diar_ref.iloc[line,len(data.columns)-1] = current_speaker
        else:
            # else fill with previous speaker
            diar_ref.iloc[line,len(data.columns)-1] = current_speaker

    # Separate start time and end time
    if len(data.columns)<3:
        diar_ref_split = diar_ref.copy()
        diar_ref_split[2] = diar_ref[1]
        split_dataset = diar_ref[0].str.split(' ', expand=True)
        diar_ref_split.iloc[:,0:2] = split_dataset.iloc[:,0:2]
        diar_ref_split = diar_ref.copy()
        diar_ref_split[2] = diar_ref[1]
        split_dataset = diar_ref[0].str.split(' ', expand=True)
        diar_ref_split.iloc[:,0:2] = split_dataset.iloc[:,0:2]
        diar_ref_split2 = diar_ref_split.copy()
        diar_ref_split2[3] = diar_ref_split[2]
        diar_ref_split2[4] = diar_ref_split[2]
        split_dataset0 = diar_ref_split[0].str.split('.', expand=True)
        diar_ref_split2.iloc[:,0:2] = split_dataset0.iloc[:,0:2]
        split_dataset1 = diar_ref_split[1].str.split('.', expand=True)
        diar_ref_split2.iloc[:,2] = split_dataset1.iloc[:,0]
        diar_ref_split2.iloc[:,3] = split_dataset1.iloc[:,1]
    else:
        diar_ref_split2 = diar_ref

    # rename columns
    if isinstance(diar_ref_split2[0][0], str):
        ground_truth_df = diar_ref_split2.rename(index=str, columns={0: "start time sec", 1: "start time msec", 2: "end time sec", 3: "end time msec", 4: "speaker"})    
    else:
        ground_truth_df = diar_ref_split2.rename(index=str, columns={0: "start time", 1: "end time", 2: "speaker"})  

    columns = ["start time sec", "end time sec", "speaker"]
    ground_truth_df_compress = pd.DataFrame(columns = columns) 

    if isinstance(ground_truth_df.iloc[0][0], str):
        start_time = int(ground_truth_df.iloc[0]["start time sec"])+int(ground_truth_df.iloc[0]["start time msec"])/100
        # Create DF with single line for each speaker talking
        for line in range(ground_truth_df.shape[0]-1):
            if ground_truth_df.iloc[line]["speaker"] != ground_truth_df.iloc[line+1]["speaker"]:
                # close current line
                end_time = int(ground_truth_df.iloc[line]["end time sec"])+int(ground_truth_df.iloc[line]["end time msec"])/100
                new_row = pd.DataFrame([[start_time, end_time, ground_truth_df.iloc[line]["speaker"]]], columns=columns)
                ground_truth_df_compress = ground_truth_df_compress.append(new_row, ignore_index=True)
                start_time = int(ground_truth_df.iloc[line+1]["start time sec"])+int(ground_truth_df.iloc[line+1]["start time msec"])/100
            if line == ground_truth_df.shape[0]-2:
                # last line, close it
                end_time = int(ground_truth_df.iloc[line+1]["end time sec"])+int(ground_truth_df.iloc[line+1]["end time msec"])/100
                new_row = pd.DataFrame([[start_time, end_time, ground_truth_df.iloc[line]["speaker"]]], columns=columns)
                ground_truth_df_compress = ground_truth_df_compress.append(new_row, ignore_index=True)  
    else:
        start_time = ground_truth_df.iloc[0]["start time"]
        # Create DF with single line for each speaker talking
        for line in range(ground_truth_df.shape[0]-1):
            if ground_truth_df.iloc[line]["speaker"] != ground_truth_df.iloc[line+1]["speaker"]:
                # close current line
                end_time = ground_truth_df.iloc[line]["end time"]
                new_row = pd.DataFrame([[start_time, end_time, ground_truth_df.iloc[line]["speaker"]]], columns=columns)
                ground_truth_df_compress = ground_truth_df_compress.append(new_row, ignore_index=True)
                start_time = ground_truth_df.iloc[line+1]["start time"]
            if line == ground_truth_df.shape[0]-2:
                # last line, close it
                end_time = ground_truth_df.iloc[line+1]["end time"]
                new_row = pd.DataFrame([[start_time, end_time, ground_truth_df.iloc[line]["speaker"]]], columns=columns)
                ground_truth_df_compress = ground_truth_df_compress.append(new_row, ignore_index=True) 

    return ground_truth_df_compress