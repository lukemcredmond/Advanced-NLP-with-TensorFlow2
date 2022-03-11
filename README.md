# Advanced-NLP-with-TensorFlow2


Code can be found at: https://github.com/PacktPublishing/Advanced-NLP-with-TensorFlow-2
Publication date: February 2021
Publisher: Packt
Pages: 380
ISBN: 9781800200937


Notes from working my way through the book

## Tools
colab.research.google.com
!pip install stanfordnlp
!pip install sklearn
!pip install gensim
## Basics


### Terminolgy
BPE ->  Byte Pair Encoding
layers
x units
relu -> model.add(tf.keras.layers.Dense(num_units, input_dim=input_dims, 
activation='relu'))
sigmoid layer
optimizers [adam]
loss
x input features
POS tagging -> part of speech tagging
Stemming and lemmatization -> removing tenses/trying to normalise words {depend : [depends, depending, depended, dependent]} they all are the same
RNN  -> Recurrent Neural Networks
LSTM -> unit for Long Short-Term Memory
BiLSTMs -> Bi-directional LSTMs
NER -> Named Entity Recognition
NLU -> Natural Language Understanding
NLP -> Natural Language Processing
CRFs -> Conditional Random Fields
ReLUs
Cell value -> or memory of the network, also referred to as the cell, which stores accumulated knowledge
GRUs -> Gated recurrent units
GMB -> 
GloVe -> Global Vectors for Word Representation
Input gate -> which controls how much of the input is used in computing the new cell value
Output gate -> which determines how much of the cell value is used in the output
Forget gate -> which determines how much of the current cell value is used for updating the cell value
overfitting -> Groningen Meaning Bank (GMB)
gazetteer -> A gazetteer is like a database of important geographical entities
IOB -> In-Other-Begin
SOTA -> (might be in #ch2)
SGD -> Stochastic Gradient Descent #ch4

Tokenization -> Reviews need to be tokenized into words.
Encoding -> These words need to be mapped to integers using the vocabulary.
Padding -> Reviews can have variable lengths, but LSTMs expect vectors of the same length. So, a constant length is chosen. Reviews shorter than this length are padded with a specific vocabulary index, usually 0 in TensorFlow. Reviews longer than this length are truncated. Fortunately, TensorFlow provides such a function out of the box.


- Text normalization
    - case normalization, 
    - tokenization and stop word removal/segmentation, 
    - Parts-of-Speech (POS) tagging, 
    - stemming

- Feature extraction
    - when the weights learned from the pre-trainning are frozen in the fine tuning layer (part of Sequential learning)
## chapters
#ch1 -> basics & build a spam detector
#ch2 -> Sentiment 
#ch3 -> Named Entity Recognition (NER) With BiLSTMs, CRFs & Viterbi Decoding
#ch4 -> Transfer Learning with BERT

# Links 
[How to get GPU working](https://spltech.co.uk/how-to-install-tensorflow-2-5-with-cuda-11-2-and-cudnn-8-1-for-windows-10/)
[available Tensorflow datasets](https://www.tensorflow.org/datasets/catalog/overview#all_datasets)