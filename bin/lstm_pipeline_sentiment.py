#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Sun Apr  2 10:33:22 2017

@author: firojalam
"""
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import numpy as np
import sys
from cnn import data_process as data_process
from cnn import cnn_filter as cnn_filter
from gensim.models import KeyedVectors
from keras.layers import Dense, Input, Dropout, Activation, Flatten
from keras.models import Sequential,Model
from keras.layers import Merge, LSTM, Dense, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Embedding
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint
import cnn.performance as performance
import os
import dnn_lex
from keras.models import load_model
import pickle
import zipfile


seed = 1337
np.random.seed(seed)


def save_model(model, model_file_name, tokenizer,label_encoder):

    model_file=model_file_name+".hdf5"
    tokenizer_file=model_file_name+".tokenizer"
    label_encoder_file=model_file_name+".label_encoder"

    configfile=model_file_name+".config"
    configFile=open(configfile,"w")
    configFile.write("model_file="+model_file+"\n")
    configFile.write("tokenizer_file="+tokenizer_file+"\n")
    configFile.write("label_encoder_file="+label_encoder_file+"\n")
    configFile.close()
    
    files=[]
    files.append(configfile)
    
    # serialize weights to HDF5
    model.save(model_file)
    files.append(model_file)
    
    # saving tokenizer
    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    files.append(tokenizer_file)
    
    # saving label_encoder
    with open(label_encoder_file, 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    files.append(label_encoder_file)

#    print 'creating archive'
#    zf = zipfile.ZipFile(model_file_name+'.zip', mode='w')    
#    for zfile in files:
#        try:
#            print ('Adding '+zfile)
#            zf.write(zfile)
#        finally:
#            print 'closing'
#    zf.close()        
#                
#    print("Saved "+model_file+'.zip'+" to disk") 
    
if __name__ == '__main__':    
    # Read train-set data
    trainFile=sys.argv[1]
    devFile=sys.argv[2]    
    tstFile=sys.argv[3]    
    resultsFile=sys.argv[4]
    outFile=open(resultsFile,"w")
    
    MAX_SEQUENCE_LENGTH = 20
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 300
    batch_size=256    
    nb_epoch=100    
    modelFile="/Users/firojalam/QCRI/w2v/GoogleNews-vectors-negative300.txt"    
    modelFile="/export/home/fialam/crisis_semi_supervised/crisis-tweets/model/crisis_word_vector.txt"    
    modelFile="../w2v_models/crisis_word_vector.txt"
    emb_model = KeyedVectors.load_word2vec_format(modelFile, binary=False)
    

    delim="\t"
    train_x,train_y,train_le,train_labels,word_index,tokenizer=data_process.getTrData(trainFile,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,delim) 

    dev_x,dev_y,dev_le,dev_labels,_=data_process.getDevData2(devFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)                
    test_x,test_y,test_le,test_labels,_=data_process.getDevData2(tstFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)
    print("Train: "+str(len(train_x)))
    
    y_true=np.argmax(train_y, axis = 1)
    y_true=train_le.inverse_transform(y_true)
    nb_classes=len(set(y_true.tolist()))
    print ("Number of classes: "+str(nb_classes))

    
    
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_matrix=dnn_lex.prepareEmbedding(word_index,modelFile)
    nb_words = min(MAX_NB_WORDS, len(word_index)+1)
    
    
    nb_filter = 250 
    filter_length = 3
    
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    
    print(MAX_SEQUENCE_LENGTH)
    nb_words = min(MAX_NB_WORDS, len(word_index)+1)    
    embedding_layer=Embedding(output_dim=EMBEDDING_DIM, input_dim=nb_words, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,trainable=False)(inputs)
    
    callback = callbacks.EarlyStopping(monitor='val_acc',patience=30,verbose=0, mode='max')        
    best_model_path="models/lstm_sentiment_best.hdf5"
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [callback,checkpoint]        
    R,C=train_x.shape

    lstm=Bidirectional(LSTM(units=256, activation='sigmoid', recurrent_activation='hard_sigmoid'))(embedding_layer)    
    network=Dense(200,activation='relu')(lstm)
    network=Dense(100,activation='relu')(network)

    out=Dense(nb_classes, activation='softmax')(network)
    model=Model(inputs=inputs,outputs=out)
    model.compile(loss='categorical_crossentropy',optimizer='Adagrad',metrics=['accuracy'])
    model.fit([train_x], train_y, batch_size=batch_size, epochs=nb_epoch,verbose=0, validation_data=([dev_x], dev_y))
    
    ###save the model
    #model.save(best_model_path)
    save_model(model,best_model_path,tokenizer,train_le)
    #model = load_model(best_model_path)


    
    dev_pred=model.predict([dev_x], batch_size=batch_size, verbose=1)
    test_pred=model.predict([test_x], batch_size=batch_size, verbose=1)
    
    ######Dev    
    accu,P,R,F1,wAUC,AUC,report=performance.performance_measure_tf(dev_y,dev_pred,dev_le,dev_labels,devFile)
    wauc=wAUC*100
    auc=AUC*100
    precision=P*100
    recall=R*100
    f1_score=F1*100
    result=str("{0:.2f}".format(auc))+"\t"+str("{0:.2f}".format(wauc))+"\t"+str("{0:.2f}".format(precision))+"\t"+str("{0:.2f}".format(recall))+"\t"+str("{0:.2f}".format(f1_score))+"\n"
    print(result)
    print (report)
    outFile.write(tstFile+"\n")
    outFile.write(result)
    outFile.write(report)
    
    ######Test
    accu,P,R,F1,wAUC,AUC,report=performance.performance_measure_tf(test_y,test_pred,test_le,test_labels,tstFile)
    wauc=wAUC*100
    auc=AUC*100
    precision=P*100
    recall=R*100
    f1_score=F1*100
    result=str("{0:.2f}".format(auc))+"\t"+str("{0:.2f}".format(wauc))+"\t"+str("{0:.2f}".format(precision))+"\t"+str("{0:.2f}".format(recall))+"\t"+str("{0:.2f}".format(f1_score))+"\n"    
    print(result)
    print (report)
    outFile.write(devFile+"\n")
    outFile.write(result)
    outFile.write(report)
        