
# SOUNDPROOF 


### Consulting project for RedCarpetUp.com done as part of Artificial Intelligence Insight Fellowship


# Background

In India, credit and finance companies are able to service less than 3% of the customer base because there are no widespread credit bureaus to profile and score customers. RedCarpet lands to customers to finance their online purchases. As they have access to private and vulnerable data, they must protect their customers information. 

The goal of the project is identifying fraud in the company's call center using a system that receives two audio call files and decides if the customer in the two calls is the same person.

# Repository Overview

The code in this repository is divided into modules, each responsible for a different part of the pipeline as can be seen in the following pipeline sketch:
![System Pipeline](https://github.com/ninashp/Speaker-Identification/blob/master/images/pipeline.jpg)

**config** folder contains default configuration

**data** folder contains samples of synthetic and open source data used in this project

**src** folder contains all source code and unitesting


# Model


This project is using an LSTM neural network for speaker identification with [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467). Model code is based on open source implementation of the speaker verification paper that can be found on GitHub [Speaker_Verification](https://github.com/Janghyun1230/Speaker_Verification)


# Data


The original data used for this project is proprietary, but I've used a several open source resources along the way: 

*   For training the LSTM Neural Network I've used the open source [CSTR VCTK Corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)
*   For validation of the diarization module I've used [Santa Barbara Corpus of Spoken American English](http://www.linguistics.ucsb.edu/research/santa-barbara-corpus), of the conversations in the dataset I've chosen those with two participants
*   To validate the whole system I've generated a synthetic dataset of conversations using the CSTR VCTK Corpus. The synthetic data can be found in the data folder

NOTE: The input data expected to be a wav format mono, PCM signed sampled at 8000Hz. 


# Dependencies

*   python 3.5+
*   numpy
*   tensorflow
*   librosa
*   scipy
*   sklearn
*   WebRTC VAD found at [https://github.com/wiseman/py-webrtcvad](https://github.com/wiseman/py-webrtcvad) 


# Execution

- To verify two calls, add the audio files into /data/calls_to_compare folder and call main.c
- To create embedding run the embeddings_db.c. The database can be created either by using separate speech utterances or extracted from phone calls according to config indication. source folder for datased for embeddings creation can be specified in the config file as well.
- To re-train the model run the main function inside LSTM speaker identification module.


# Results

To get indicative results on a large dataset I’ve created synthetic dataset of a artificial call center calls with several operators and sufficient amount of customers. I did this by stitching random sentences of different speakers obtained from the CSTR VCTK Corpus dataset.

The confusion matrix below was achieved on the synthetic data.
The interesting metric is the recall that represents the amount of fraudulent calls I was able to detect. For the data below the recall is 93%

<table>
  <tr>
   <td><strong>Synthetic Data</strong>
   </td>
   <td>Predicted: 
<p>
NOT SAME USER
   </td>
   <td>Predicted: 
<p>
SAME USER
   </td>
  </tr>
  <tr>
   <td>Actual: 
<p>
NOT SAME USER
   </td>
   <td><strong>TP:</strong> 716
   </td>
   <td><strong>FN:</strong> 54
   </td>
  </tr>
  <tr>
   <td>Actual: 
<p>
SAME USER
   </td>
   <td><strong>FP:</strong> 45
   </td>
   <td><strong>TN:</strong> 525
</table>

### Recall: 93%
### F1: 0.93 
