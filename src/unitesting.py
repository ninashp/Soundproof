"""
This file contains unit testing for all the modules of the package
""" 
import os
import signal_processing
from signal_processing import SignalProcessing
from configuration import get_config
import speaker_diarization
import datetime
import librosa
import numpy as np
import speaker_verfier
import embeddings_db
import speaker_verification_lstm_model
import shutil
import pandas as pd
import utils

config = get_config()   # get arguments from parser

def unitest_create_embeddings():
    embeddings_db.create_embeddings_db_from_calls(is_test=1)
    embeddings_db.create_embeddings_db_from_utternaces(is_test=1)
    return 1

def unitest_signal_processing__call_type():
    # ground truth labels for call types
    call_types_ground_truth = {
        "f8018973-ddda-4f0a-9954-e8ceed4c0154.wav" : SignalProcessing.FIRST_SPEAKER_REPRESENTATIVE,
        "f48a95e0-e000-40aa-aec0-6d11ef07edd3.wav" : SignalProcessing.FIRST_SPEAKER_REPRESENTATIVE,
        "bf9f10ec-c868-482c-8199-9e412aa4237a.wav" : SignalProcessing.FIRST_SPEAKER_REPRESENTATIVE,
        "980158b4-0e2d-4309-9208-38c5e293fe19.wav" : SignalProcessing.FIRST_SPEAKER_REPRESENTATIVE,
        "22bcbfea-dde1-4df1-841a-a1d2840633d7_0_1_r.wav" : SignalProcessing.FIRST_SPEAKER_CUSTOMER,
        "10e3ed40-8f1f-4a0f-a8f8-1dcaa2954ef1.wav" : SignalProcessing.FIRST_SPEAKER_REPRESENTATIVE,
        "3d1ab4ad-0788-4514-b039-47582a4e27b0.wav" : SignalProcessing.FIRST_SPEAKER_REPRESENTATIVE,
        "d6088637-61c3-4d4a-92ec-27fa557368e3.wav" : SignalProcessing.FIRST_SPEAKER_REPRESENTATIVE,
        "f7b818f0-c61c-4bac-bd46-ec897bf37652.wav" : SignalProcessing.FIRST_SPEAKER_REPRESENTATIVE,
        "d8b9f4ed-d06d-4f1b-84c8-5d24d3d7e671.wav" : SignalProcessing.FIRST_SPEAKER_REPRESENTATIVE,
    }

    success_count = 0
    for single_call_file in os.listdir(config.calls_path):
        call_type = SignalProcessing.identify_call_type(os.path.join(config.calls_path,single_call_file))
        print("call path:"+single_call_file)
        SignalProcessing.print_call_type(call_type)
        if call_type == call_types_ground_truth[single_call_file]:
            print("PASS")
            success_count += 1
        else:
            print("FAIL")
    success_rate = success_count/len(os.listdir(config.calls_path))
    print("success rate of call type test: "+str(success_rate))
    return success_rate 

def unitest_two_person_diarization_split_single_call(audio_path, ground_truth_path = "", target_path = ""):
    timestamp_dict, _ = speaker_diarization.two_person_diarization(audio_path)

    if target_path:
        # save sound to file 
        call_audio, _  = librosa.core.load(audio_path, config.sr)
        first_speaker_sound = []
        second_speaker_sound = []
        for timestamp in timestamp_dict["FIRST SPEAKER"]:
            first_speaker_sound = np.append(first_speaker_sound, 
                                            call_audio[librosa.core.time_to_samples(timestamp[0], config.sr) :  
                                                        librosa.core.time_to_samples(timestamp[1], config.sr)])
        for timestamp in timestamp_dict["SECOND SPEAKER"]:
                second_speaker_sound = np.append(second_speaker_sound, 
                                            call_audio[librosa.core.time_to_samples(timestamp[0], config.sr) :  
                                                        librosa.core.time_to_samples(timestamp[1], config.sr)])                   
        librosa.output.write_wav(target_path+"speaker_0.wav", first_speaker_sound, config.sr)
        librosa.output.write_wav(target_path+"speaker_1.wav", second_speaker_sound, config.sr)

    if ground_truth_path:
        # Validate timestamps using pandas DF ground truth
        ground_truth_df = utils.get_pandas_ds_from_trn(ground_truth_path)
        ground_truth_start_times = list(map(float, ground_truth_df['start time sec'].values))
        ground_truth_end_times = list(map(float, ground_truth_df['end time sec'].values))
        total_duration = 0
        success_duration = 0
        # iterate over the two speakers timestamps and validate with ground truth
        for speaker_str, timestamps in timestamp_dict.items():
            speaker = 0 if speaker_str == "FIRST SPEAKER" else 1
            # iterate over each segment to get the durration of correct split
            for timestamp in timestamps:
                start_time = timestamp[0]
                end_time = timestamp[1]
                # get total duration detected
                total_duration += timestamp[1]-timestamp[0]
                # look in ground truth to find the correct speaker in this segment
                for idx, start_time_gt in enumerate(ground_truth_start_times):
                    end_time_gt = ground_truth_end_times[idx]
                    if start_time_gt > start_time:
                        # segment starts in the middle, it's ok
                        start_time = start_time_gt
                    if (start_time_gt <= start_time) and (end_time_gt > start_time):
                        if end_time_gt > end_time:
                            # all segment is inside segment of one speaker, check if he is correct
                            if ground_truth_df.iloc[idx]['speaker'] == speaker:
                                success_duration += end_time - start_time
                            break
                        else:
                            # split into chunks
                            # new start time is the end time of current chunk
                            # all segment is inside segment of one speaker, check if he is correct
                            if ground_truth_df.iloc[idx]['speaker'] == speaker:
                                success_duration += end_time_gt - start_time           
                            # next iteration start from the end of current segment
                            start_time = end_time_gt
                            continue 
                else:
                    print("error: cant find time")
        success_rate = success_duration/total_duration
        return success_rate
    else:
        return 0

def unitest_two_person_diarization_split():
    files_list = os.listdir(config.diarization_test_path)

    for file_name in files_list:
        # skip files which are not wav
        if file_name[-4:] != ".wav":
            continue
        audio_path = os.path.join(config.diarization_test_path, file_name)
        ground_truth_path = audio_path[:-4] + ".trn"
        target_path = os.path.join(config.diarization_test_path, file_name[:-4]+"_diarization.wav")
        print("processing",audio_path,"....")
        success_rate = unitest_two_person_diarization_split_single_call(audio_path, ground_truth_path=ground_truth_path, target_path=target_path)
        print("Done, success rate:", success_rate)

def unitest_two_person_diarization_split_real_call():
    test_files_folder = config.calls_to_compare_path
    target_dir = "../data/diarization"

    for test_files_name in os.listdir(test_files_folder):
        audio_path = os.path.join(test_files_folder, test_files_name)
        if os.path.isfile(audio_path):
            print("processing",audio_path,"....")
            success_rate = unitest_two_person_diarization_split_single_call(audio_path, target_path=os.path.join(target_dir, test_files_name))
            print("Done, success rate:", success_rate)
    return 1

def unitest_speaker_verification_sanity():
    # as sanity check, compare the same file to itself
    calls_to_compare_path = "../data/speaker_verification"
    is_same_speaker, similarity_matrix = speaker_verfier.speaker_verifier(calls_to_compare_path)
    if not is_same_speaker:
        return 0
    else:
        # check similarity matrix, value in first entry should be equal to 1
        if 1-similarity_matrix[0][1] < 1e-5:
            return 1
        else:
            return 0

def speaker_verifier_single_speaker(path_to_speaker_utterances):
    """ Check if in two utterances of the same speaker are classified as the same 
    """   
    # split into two folders
    test1_filder = os.path.join(path_to_speaker_utterances,"test__1")
    test2_filder = os.path.join(path_to_speaker_utterances,"test__2")
    os.makedirs(test1_filder)
    os.makedirs(test2_filder)

    for idx, utter_file in enumerate(os.listdir(path_to_speaker_utterances)):
        if os.path.isfile(os.path.join(path_to_speaker_utterances,utter_file)):
            if idx % 2 == 0:
                shutil.copyfile(os.path.join(path_to_speaker_utterances,utter_file), os.path.join(test1_filder,utter_file))
            else:
                shutil.copyfile(os.path.join(path_to_speaker_utterances,utter_file), os.path.join(test2_filder,utter_file))

    # Get embeddings of the utterances in test1
    single_speaker_frames1 = embeddings_db.get_single_speaker_frames_from_utterances_path(test1_filder)
    single_speaker_embeddings1 = speaker_verification_lstm_model.extract_embedding(single_speaker_frames1)

    # Get embeddings of the utterances in test2
    single_speaker_frames2 = embeddings_db.get_single_speaker_frames_from_utterances_path(test2_filder)
    single_speaker_embeddings2 = speaker_verification_lstm_model.extract_embedding(single_speaker_frames2)

    # remove temp directories
    shutil.rmtree(test1_filder) 
    shutil.rmtree(test2_filder) 

    # create two mean embeddings
    single_speaker_embeddings = np.append([np.mean(single_speaker_embeddings1, axis=0)], [np.mean(single_speaker_embeddings2, axis=0)], axis=0)

    # append extracted embeddings to the other embedding in the DB
    embeddings_db_npy = np.load(os.path.join(config.embeddings_db_path, "embedding_db.npy"))
    united_embeddings_array = np.append(single_speaker_embeddings, embeddings_db_npy, axis=0)
    # embedding extracted, using similarity matrix to verify the user
    similarity_matrix = speaker_verification_lstm_model.create_similarity_matrix(united_embeddings_array)

    print(similarity_matrix[0:2])
    # first line of the similarity matrix is first embedding compared to all the others
    # we need to check if it's highest similarity is with the second value
    # second line is the second embedding compared to all the others
    # we need to check if it's highest similarity with the first
    similarity_vector1 = similarity_matrix[0][1:]
    similarity_vector2 = np.append(similarity_matrix[1][0],similarity_matrix[1][2:])
    if np.argmax(similarity_vector1) == 0 and np.argmax(similarity_vector2) == 0:
        return 1, similarity_matrix[0:2]
    else:
        # we might have this customer in the embedding db, so if both embedding are similar to the same one, return true
        if np.argmax(similarity_vector1) == np.argmax(similarity_vector2):
            return 1
        else:
            return 0

def unitest_speaker_verifier_single_speaker():
    """ Test same speaker different utterances to check if the same speaker can be recognized
    """
    path_to_speakers_folder = "../data/single_speaker_uterrances"
    success_count = 0
    for single_speaker_file in os.listdir(path_to_speakers_folder):
        print("--- testing "+single_speaker_file)
        if speaker_verifier_single_speaker(os.path.join(path_to_speakers_folder, single_speaker_file)):
            print("result: same user!")
            success_count += 1
        else:
            print("result: different user!")
    success_rate = success_count/len(os.listdir(path_to_speakers_folder))
    print("success rate of speaker verifier single speaker: "+str(success_rate))
    return success_rate

def unitest_speaker_verification():
    # compare two different parts of the same file
    calls_to_compare_path = "../data/speaker_verification"
    is_same_speaker, _ = speaker_verfier.speaker_verifier(calls_to_compare_path)
    return is_same_speaker

def build_confusion_matrix():
    """ Construct confusion matrix.
    """
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0

    # Test same speakers
    test_files_folder = "../data/synthetic_call_center/synthetic_calls_positives"
    files_list = []
    for test_files_name in os.listdir(test_files_folder):
        audio_path = os.path.join(test_files_folder, test_files_name)
        if os.path.isfile(audio_path):
            if test_files_name.find("_1") != -1:
                files_list.append(test_files_name)
    files_list = list(set(files_list))
    
    # run identification on each pair of calls
    for file_path in files_list:
        first_file = file_path
        second_file = file_path.replace("_1", "_2")       
        print("processing",first_file,"and",second_file)
        # Create temporary directory and copy the files there
        temp_dir_path = os.path.join(test_files_folder,"temp")
        os.makedirs(temp_dir_path)
        shutil.copyfile(os.path.join(test_files_folder,first_file), os.path.join(temp_dir_path,first_file))
        shutil.copyfile(os.path.join(test_files_folder,second_file), os.path.join(temp_dir_path,second_file))
        # run speaker verifier to check if speaker is the same
        is_same_speaker, _ = speaker_verfier.speaker_verifier(temp_dir_path)

        # TEMP - output diarization result
        target_path1 = os.path.join(temp_dir_path, first_file)
        target_path2 = os.path.join(temp_dir_path, second_file)
        unitest_two_person_diarization_split_single_call(os.path.join(test_files_folder,first_file), ground_truth_path = "", target_path = target_path1)
        unitest_two_person_diarization_split_single_call(os.path.join(test_files_folder,second_file), ground_truth_path = "", target_path = target_path2)

        print("Done, same speaker result is:", "true" if is_same_speaker else "false")
        # remove temp directory
        shutil.rmtree(temp_dir_path)
        # fill matrix
        if is_same_speaker:
            true_positives += 1
        else:
            false_negatives +=1

    # Now different speakers
    test_files_folder = "../data/synthetic_call_center/synthetic_calls_negatives"

    files_list = []
    for test_files_name in os.listdir(test_files_folder):
        audio_path = os.path.join(test_files_folder, test_files_name)
        if os.path.isfile(audio_path):
            files_list.append(test_files_name)

    # run identification on each pair of calls
    for idx, first_file in enumerate(files_list):
        print(idx,"--- out of",len(files_list))
        for second_file in files_list[idx+1:]:
            print("processing",first_file,"and",second_file)
            # Create temporary directory and copy the files there
            temp_dir_path = os.path.join(test_files_folder,"temp")
            os.makedirs(temp_dir_path)
            shutil.copyfile(os.path.join(test_files_folder,first_file), os.path.join(temp_dir_path,first_file))
            shutil.copyfile(os.path.join(test_files_folder,second_file), os.path.join(temp_dir_path,second_file))
            # run speaker verifier to check if speaker is the same
            is_same_speaker, _ = speaker_verfier.speaker_verifier(temp_dir_path)
            print("Done, same speaker result is:", "true" if is_same_speaker else "false")
            # remove temp directory
            shutil.rmtree(temp_dir_path)
            # fill matrix
            if is_same_speaker:
                false_positives += 1
            else:
                true_negatives +=1

    print("*****Confision matrix:******")
    print("True positives:",true_positives)
    print("False positives:",false_positives)
    print("False negatives:",false_negatives)
    print("True negatives:",true_negatives)

    print("precision:",true_positives/(true_positives+false_positives))
    print("recall:",true_positives/(true_positives+false_negatives))

    return 1

if __name__ == "__main__":
    # run all unit tests 
    if unitest_create_embeddings() != 1:
        print("**** unitest_create_embeddings FAILED!!!")
    else:
        print("**** unitest_create_embeddings PASSED!!!")
    if unitest_signal_processing__call_type() != 1:
        print("**** unitest_signal_processing__call_type FAILED!!!")
    else:
        print("**** unitest_signal_processing__call_type PASSED") 
    if unitest_two_person_diarization_split() > 0.9:
        print("**** unitest_two_person_diarization_split FAILED!!!")
    else:
        print("**** unitest_two_person_diarization_split PASSED")
    if unitest_two_person_diarization_split_real_call() != 1:
        print("**** unitest_two_person_diarization_split_real_call FAILED!!!")
    else:
        print("**** unitest_two_person_diarization_split_real_call PASSED")
    if unitest_speaker_verification_sanity() != 1:
        print("**** unitest_speaker_verification_sanity FAILED!!!")
    else:
        print("**** unitest_speaker_verification_sanity PASSED")
    if unitest_speaker_verifier_single_speaker() != 1:
        print("**** unitest_speaker_verifier_single_speaker FAILED!!!")
    else:
        print("**** unitest_speaker_verifier_single_speaker PASSED")   
    if unitest_speaker_verification() != 1:
        print("**** unitest_speaker_verification FAILED!!!")
    else:
        print("**** unitest_speaker_verification PASSED")
    if build_confusion_matrix() != 1:
        print("**** build_confusion_matrix FAILED!!!")
    else:
        print("**** build_confusion_matrix PASSED")


        