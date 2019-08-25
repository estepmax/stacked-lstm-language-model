from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import numpy as np

import string 

tokenizer = Tokenizer(lower=True)

def load_corpus(filename):
    
    file = open("./utilities/{}".format(filename),"r")
    corpus = file.read()
    file.close()
    
    return corpus

def load_dataset():
    return np.load('./utilities/data.npz')

def create_dataset(corpus,sep,save=True):
    
    sentences = corpus.split(sep)

    tokenizer.fit_on_texts(sentences)

    vocab_size = len(tokenizer.word_index) + 1
    
    sequences = tokenizer.texts_to_sequences(sentences)
    sequences = pad_sequences(sequences,padding="pre",maxlen=vocab_size)
    
    predictors,labels = sequences[:,:-1],sequences[:,-1]
    labels = to_categorical(labels,num_classes=vocab_size)

    sequence_length = predictors.shape[1]
    
    if save:
        print('Saving generated sentence vectors to disk...')
        np.savez('./utilities/data',predictors=predictors,labels=labels,vocab_size=vocab_size,sequence_length=sequence_length)
    
    return predictors,labels,vocab_size,sequence_length

