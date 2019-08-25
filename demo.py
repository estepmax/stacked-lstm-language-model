from tgmodel.model import TGModel

from utilities import data

def main():

    corpus = data.load_corpus("some-random-corpus.txt")
    predictors,labels,vocab_size,sequence_length = data.create_dataset(corpus,"\n",save=False)    

    model = TGModel()
    
    model.initialize(vocab_size,sequence_length)
    model.fit(predictors,labels)

    seed = ""
    sentence_length = "enter-your-desired-sentence-length"

    model.predict(seed,sentence_length,sequence_length,data.tokenizer)
   
if __name__=='__main__':
    main()
    
