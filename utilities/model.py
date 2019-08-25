from keras.callbacks import EarlyStopping

def initialize():
    settings = {  
        "return_sequences" : True,
        "activation_0" : "relu",
        "activation_1" : "softmax",  
        "output_dim" : 200,
        "lstm_units" : 200,  
        "dense_units" : 100,
        "rate" : 0.2
    } 
    return settings

def fit():
    settings = {  
        "epochs" : 100,
        "verbose" : True,   
        "batch_size" : 128  
        #"callbacks" : [EarlyStopping(monitor="loss",min_delta=0)]      
    } 
    return settings

def compile():
    settings = {
        "loss" : "categorical_crossentropy",
        "optimizer" : "rmsprop",
        "metrics" : ["accuracy"]
    }
    return settings 
