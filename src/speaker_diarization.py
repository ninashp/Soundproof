""" 
This file contains the code for speaker diarization module.

"""
import librosa
import numpy as np
from sklearn.cluster import KMeans
from configuration import get_config
import utils
import signal_processing
from signal_processing import SignalProcessing
from VAD_segments import VAD_chunk
import speaker_verification_lstm_model
import scipy
import statistics

# get arguments from parser
config = get_config()  

def two_person_diarization(call_file):
    """ Diarization of a call of 2 speakers.
        Input: path to a call
        Output: a dictionary of two np arrays of timestamps in seconds - one per speaker.
    """
    # create embeddings and cluster them
    seg_times, speech_segs = VAD_chunk(2, call_file)
    assert speech_segs != [],"No voice apctivity detected."

    all_embeddings = []
    embedding_times = []
    for speech_seg, seg_time in zip(speech_segs, seg_times):
        STFT_frames = SignalProcessing.split_segment_to_frames(speech_seg)
        if not STFT_frames:
            # not enough frames, continue to next segment
            continue
        STFT_frames = np.stack(STFT_frames, axis=1)
        embeddings = speaker_verification_lstm_model.extract_embedding(STFT_frames)
        # calculate time stamps for each embedding
        delta_t = (seg_time[1]-seg_time[0]) / embeddings.shape[0]
        times_start = np.linspace(seg_time[0], seg_time[1], num=embeddings.shape[0])
        times_end = np.linspace(seg_time[0]+delta_t, seg_time[1]+delta_t, num=embeddings.shape[0])
        for idx, embedding in enumerate(embeddings):
            all_embeddings.append(embedding)
            embedding_times.append([times_start[idx], times_end[idx]])

    # Using K-Means to separate the two speakers embeddings
    kmeans_emb = KMeans(n_clusters=2, random_state=0).fit(all_embeddings)
    # Getting the cluster labels
    labels = kmeans_emb.predict(all_embeddings)
    # Taking only the embeddings that are close to the centers 
    distances_from_centers = []
    distances_from_centers.append([])
    distances_from_centers.append([])
    for embedding, label in zip(all_embeddings, labels):
        distances_from_centers[label].append(scipy.spatial.distance.euclidean(embedding, kmeans_emb.cluster_centers_[label]))
    # Take only the most certain segments
    median_dist = []
    median_dist.append(statistics.median(distances_from_centers[0]))
    median_dist.append(statistics.median(distances_from_centers[1]))
    # Create list of times and embedding per speaker
    first_speaker_label = labels[0]
    speaker0_times = []
    speaker1_times = []
    speaker0_embeddings = []
    speaker1_embeddings = []
    for idx, label in enumerate(labels):
        if scipy.spatial.distance.euclidean(all_embeddings[idx], kmeans_emb.cluster_centers_[label]) < median_dist[label]:
            if label == first_speaker_label:
                speaker0_times.append(embedding_times[idx])
                speaker0_embeddings.append(all_embeddings[idx])
            else:
                speaker1_times.append(embedding_times[idx])
                speaker1_embeddings.append(all_embeddings[idx])

    # build dictionary to return
    return {"FIRST SPEAKER":speaker0_times, "SECOND SPEAKER":speaker1_times}, {"FIRST SPEAKER":speaker0_embeddings, "SECOND SPEAKER":speaker1_embeddings}
