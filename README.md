
# Speaker identification 


### Consulting project for RedCarpetUp.com done as Insight Fellowship Project


# Motivation



*   RedCarpetUp lends to customers in India to finance their online purchases 


*   They need a caller identification system to identify if the caller is the same person he claims he is


# Project


  The product is a caller identification system that will be integrated into the internal caller system of RedCarpetUp 


# Model



*   The model will be RNN built using GE2E loss function
*   Paper: [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467) 
*   Code references: [Speaker_Verification,](https://github.com/Janghyun1230/Speaker_Verification) [uis-rnn](https://github.com/google/uis-rnn)
*   Trained on AWS/CloudML


# Data



*   The data is calls from the company call center recordings 
*   Organized as Amazon S3 indexed in a postgresql database
*   Data is not tagged
*   The calls are multilingual


# Setup and installation instructions

TBD
