#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 10:24:54 2017

@author: firojalam
"""




import sys
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import load_model
import optparse
import datetime

class Instance(object):
    def __init__(self,id=1,date="",txtOrg="",txt=""):
        self.id = id
        self.date = date        
        self.txtOrg = txtOrg        
        self.txt = txt

        
def getData(dataFile,tokenizer,MAX_SEQUENCE_LENGTH,delim):
    """
    Prepare the data
    """
    data=[]
    instances=[]
    with open(dataFile, 'rU') as f:
        next(f)    
        for line in f:
            line = line.strip()   
            if (line==""):
                continue                            		
            row=line.split(delim)
            ID = row[0].strip()        
            date = row[1].strip()        
            txtOrg = row[2].strip()                    
            txt = row[3].strip()

            inst = Instance(id=ID,date=date,txtOrg=txtOrg,txt=txt)
            if(len(txt)<1):
                print txt
                continue
            data.append(txt)
            instances.append(inst)
            
    sequences = tokenizer.texts_to_sequences(data)   
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return data,instances
    
    
def load_nn_model(model_file):
    loaded_model = load_model(model_file)    
    print("Loaded model from disk")
    return loaded_model
            


def readConfig(configfile):
    configdict={}
    with open(configfile, 'rU') as f:
        for line in f:
            line = line.strip()   
            if (line==""):
                continue                            		
            row=line.split("=")
            configdict[row[0]]=row[1]
    return configdict
    
        

def write2File(outfilename, prediction,probabilities,instances):
    textFile=open(outfilename,"w")   
    textFile.write("TweetID\tDate\tSentimentLabel\tSentimentConfidence\n");    
    for lab, prob,inst in zip(prediction,probabilities,instances):
        tmpData=str(inst.id)+"\t"+inst.date+"\t"+lab+"\t"+str(prob)
        textFile.write(tmpData+"\n");        
    textFile.close
    
if __name__ == '__main__':    
    
    parser = optparse.OptionParser()
    parser.add_option('-c', action="store", dest="configfile")
    parser.add_option('-d', action="store", dest="dataFile")    
    parser.add_option('-o', action="store", dest="classifiedFile")    
    options, args = parser.parse_args()
    
    configfile=options.configfile
    dataFile=options.dataFile
    classifiedFile=options.classifiedFile
    
    delim="\t"
    MAX_SEQUENCE_LENGTH = 20
    batch_size=256    
    
    
    configdict=readConfig(configfile)
    
    loaded_model=load_nn_model(configdict["model_file"])
    tokenizer_file=configdict["tokenizer_file"]
    label_encoder_file=configdict["label_encoder_file"]    


    # loading tokenizer
    with open(tokenizer_file, 'rb') as handle:
        tokenizer = pickle.load(handle)
   
    # loading label_encoder
    with open(label_encoder_file, 'rb') as handle:
        label_encoder = pickle.load(handle)
        
                
    data,instances=getData(dataFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)                     
    
    # classify data
    a = datetime.datetime.now().replace(microsecond=0)
    prediction=loaded_model.predict([data], batch_size=batch_size, verbose=1)
    
    probability_index=np.argmax(prediction, axis = 1)
    probabilities=[]
    for index,prob in zip(probability_index,prediction): 
	#print i, prob, prob[i]
	probabilities.append(prob[index])

    class_labels=label_encoder.inverse_transform(probability_index)    
    write2File(classifiedFile,class_labels,probabilities,instances)
    
    b = datetime.datetime.now().replace(microsecond=0)
    print("Time taken: "+str((b-a)))
    