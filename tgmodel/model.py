from numpy import unique

import pandas as pd

from keras import backend as k
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.models import Model,Sequential

from keras.preprocessing.sequence import pad_sequences

from utilities import model

class TGModel(object):
    
    NAME = "TGModel"

    def __init__(self):  
                
        self.model = None
        self.history = None
        self.init_settings = model.initialize()
        self.compile_settings = model.compile()
        self.fit_settings = model.fit()

    def initialize(self,vocab_size,sequence_length):
        
        model = Sequential()

        model.add(Embedding(input_dim=vocab_size,output_dim=self.init_settings["output_dim"],input_length=sequence_length))
        
        model.add(LSTM(units=self.init_settings["lstm_units"],return_sequences=self.init_settings["return_sequences"]))
        model.add(LSTM(units=self.init_settings["lstm_units"]))
        
        model.add(Dropout(rate=self.init_settings["rate"]))
        
        model.add(Dense(units=self.init_settings["dense_units"],activation=self.init_settings["activation_0"]))
        model.add(Dense(units=vocab_size,activation=self.init_settings["activation_1"]))
        
        model.compile(**self.compile_settings)
        
        print(model.summary())

        self.model = model
    
    def fit(self,predictors,label):
        self.history = self.model.fit(predictors,label,batch_size=self.fit_settings["batch_size"],epochs=self.fit_settings["epochs"],verbose=self.fit_settings["verbose"]) 

    def predict(self,seed,sentence_length,sequence_length,tokenizer):
        
        word_dictionary = dict((v,k) for k,v in tokenizer.word_index.items())
    
        for _ in range(sentence_length):
            
            tokens = tokenizer.texts_to_sequences([seed])
            tokens = pad_sequences(tokens,maxlen=sequence_length,padding='pre')
            
            predicted = self.model.predict_classes(tokens,verbose=0)[0]

            next_word = word_dictionary[predicted]

            seed += " " + next_word

        print(seed) 

    def save_weights(self):
        print('saving model weights to disk....')
        self.model.save_weights("{}/trained/model.hd5".format(TGModel.NAME))
        print('completed')

    def load_weights(self):
        print('loading model weights from disk...')
        self.model.load_weights("{}/trained/model.hd5".format(TGModel.NAME))
        print('completed')
