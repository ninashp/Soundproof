""" 
This file contains the code for speaker verifier application.

"""

import signal_processing
import speaker_diarization
from configuration import get_config
import os
import numpy as np
from signal_processing import SignalProcessing
import speaker_verification_lstm_model

# get arguments from parser
config = get_config()

def get_single_speaker_embeddings_from_call_center_call(call_file_path):
    """ Extract embeddings of speach from a call
        Input: path to file with call audio file
        Output: embeddings for the customer in the call
    """    
    # check if this call is starts with representative or customer
    call_type = SignalProcessing.identify_call_type(call_file_path)    
    # split the call into two callers
    _, embeddings_dict = speaker_diarization.two_person_diarization(call_file_path)
    # choose customer timestamps
    if call_type == SignalProcessing.FIRST_SPEAKER_CUSTOMER:
        print(call_file_path,": FIRST_SPEAKER_CUSTOMER")
        embedding_per_call = embeddings_dict["FIRST SPEAKER"]
    else:
        print(call_file_path,": FIRST_SPEAKER_REPRESENTATIVE")
        embedding_per_call = embeddings_dict["SECOND SPEAKER"]
    # Find mean embedding per customer
    return np.mean(embedding_per_call, axis=0)

def speaker_verifier(calls_to_compare_path):
    """ Check if in two calls in a call center the customer is the same
        Intup: path to the calls if given by config 
        Output: Boolean indication of if the customers are the same  
    """   
    embeddings_db = np.load(os.path.join(config.embeddings_db_path, "embedding_db.npy"))

    # for each one of the two calls extrace the utterances of the customer side
    embeddings = []
    for single_call_file in os.listdir(calls_to_compare_path):
        call_file_path = os.path.join(calls_to_compare_path, single_call_file)
        # Extract embedding for customer
        single_speaker_embedding = get_single_speaker_embeddings_from_call_center_call(call_file_path)
        embeddings.append(single_speaker_embedding)
    # append extracted embeddings to the other embedding in the DB
    united_embeddings_array = np.append(np.asarray(embeddings),embeddings_db, axis=0)
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
            return 1, similarity_matrix[0:2]
        else:
            return 0, similarity_matrix[0:2]
        
