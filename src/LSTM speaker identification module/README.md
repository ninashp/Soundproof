## ****************************************************************************************
## This code is slightly modified from https://github.com/Janghyun1230/Speaker_Verification
## ****************************************************************************************

# Speaker_Verification
Tensorflow implementation of generalized end-to-end loss for speaker verification

### Explanation
- This code is the implementation of generalized end-to-end loss for speaker verification (https://arxiv.org/abs/1710.10467)

### Speaker Verification
- Speaker verification task is 1-1 check for the specific enrolled voice and the new voice. This task needs higher accuracy than speaker identification which is N-1 check for N enrolled voices and a new voice. 
- There are two types of speaker verification. 1) Text dependent speaker verification (TD-SV). 2) Text independent speaker verification (TI-SV). 
For caller identification only the TI-SV type is used

### Files
- configuration.py  
Argument parsing  

- data_preprocess.py  
Extract noise and perform STFT for raw audio. For each raw audio, voice activity detection is performed by using librosa library.

- utils.py   
Containing various functions for training and test.  

- model.py  
Containing train and test function. Train fucntion draws graph, starts training and saves the model and history. Test function load 
variables and test performance with test dataset.  

- main.py  
When this file is implemented, training or test begins.

