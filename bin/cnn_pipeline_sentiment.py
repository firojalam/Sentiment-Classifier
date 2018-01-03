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
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint
import cnn.performance as performance
import os
from keras.models import model_from_yaml
import pickle
import zipfile
from keras.models import load_model


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
    #modelFile="/Users/firojalam/QCRI/w2v/GoogleNews-vectors-negative300.txt"    
    modelFile="/export/home/fialam/crisis_semi_supervised/crisis-tweets/model/crisis_word_vector.txt"    
    modelFile="../w2v_models/crisis_word_vector.txt"  
    emb_model = KeyedVectors.load_word2vec_format(modelFile, binary=False)
    
    #emb_model =""
    total_train_inst = sum(1 for line in open(trainFile, 'rU'))-1
    delim="\t"
    train_x,train_y,train_le,train_labels,word_index,tokenizer=data_process.getTrData(trainFile,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,delim) 

    dev_x,dev_y,dev_le,dev_labels,_=data_process.getDevData2(devFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)                
    test_x,test_y,test_le,test_labels,_=data_process.getDevData2(tstFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)
    print("Train: "+str(len(train_x)))
    
    y_true=np.argmax(train_y, axis = 1)
    y_true=train_le.inverse_transform(y_true)
    nb_classes=len(set(y_true.tolist()))
    print ("Number of classes: "+str(nb_classes))
    
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    cnn = cnn_filter.text_cnn(emb_model,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,inputs)        
    
    callback = callbacks.EarlyStopping(monitor='val_acc',patience=30,verbose=0, mode='max')        
    best_model_path="models/cnn_sentiment_best.hdf5"
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [callback,checkpoint]        
    R,C=train_x.shape

    network=Dense(512)(cnn)
    network=Activation('relu')(network)
    network=Dense(200)(network)
    network=Activation('relu')(network)
    network=Dropout(0.02)(network)
    out=Dense(nb_classes, activation='softmax')(network)
    model=Model(inputs=inputs,outputs=out)
    model.compile(loss='categorical_crossentropy',optimizer='Adagrad',metrics=['accuracy'])
    model.fit([train_x], train_y, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=([dev_x], dev_y),callbacks=callbacks_list)

    ###save the model
    #model.save(best_model_path)
    save_model(model,best_model_path,tokenizer,train_le)
    model = load_model(best_model_path)
    
    dev_prob=model.predict([dev_x], batch_size=batch_size, verbose=1)
    test_prob=model.predict([test_x], batch_size=batch_size, verbose=1)
    
    ######Dev    
    #y_prob = model.predict_proba(dev_x,batch_size=32, verbose=1)
    accu,P,R,F1,wAUC,AUC,report=performance.performance_measure_tf(dev_y,dev_prob,dev_le,dev_labels,devFile)
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
    #y_prob = model.predict_proba(test_x,batch_size=32, verbose=1)
    accu,P,R,F1,wAUC,AUC,report=performance.performance_measure_tf(test_y,test_prob,test_le,test_labels,tstFile)
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
    