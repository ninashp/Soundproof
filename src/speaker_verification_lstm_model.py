""" 
This file contains the LSTM Deep Learning NN model usage for speaker identification in a phone call.
An existing pre-trained model is loaded from configuration path and used to retrieve embedding from speach utterances.
"""
import tensorflow as tf
import numpy as np
import os
import time
from configuration import get_config
from tensorflow.contrib import rnn
import scipy

config = get_config()

#L2 Normalization
def normalize(x):
    """ normalize the last dimension vector of the input matrix
        Input: vector
        Output: normalized vector
    """
    return x/tf.sqrt(tf.reduce_sum(x**2, axis=-1, keepdims=True)+1e-6)

def extract_embedding(data_frames):
    """ Extract embedding for single call by using pre-trained LSTM model
        Input: numpy array of data frames
        Output: numpy array representing the embedding of the speech file
    """
    tf.reset_default_graph()

    # draw graph
    input_spectrograms = tf.placeholder(shape=[config.tisv_frame, None, config.mel_nof], dtype=tf.float32) # enrollment batch (time x batch x n_mel)

    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        # Combine LSTM cells into full network 
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=input_spectrograms, dtype=tf.float32, time_major=True) 
        # the last ouput is the embedded d-vector
        embedded = normalize(outputs[-1])                 

    saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # load model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(config.model_path, "Check_Point"))
        ckpt_list = ckpt.all_model_checkpoint_paths
        loaded = 0
        for model in ckpt_list:
            # find ckpt file which matches configuration model number
            if config.model_num == int(model[-1]):    
                loaded = 1
                # restore variables from selected ckpt file
                saver.restore(sess, model)  
                break

        if loaded == 0:
            raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

        # reshaping
        data_frames_reshaped = np.reshape(data_frames, [config.tisv_frame, -1, config.mel_nof])
        embeddings_vector = sess.run(embedded, feed_dict={input_spectrograms:data_frames_reshaped})
        np.set_printoptions(precision=2)
    return embeddings_vector

def create_similarity_matrix(tested_embeddings):
    """ Callculate similarity matrix for embeddings using cosine similarity
        Compare to existing customers embeddings database
        Input: Embedding as numpy array
        Output: Similarity matrix
    """
    return 1-scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(tested_embeddings, 'cosine'))
