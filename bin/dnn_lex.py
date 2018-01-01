# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 20:42:24 2017

@author: firojalam
"""


import numpy as np
np.random.seed(1337)  # for reproducibility

#from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys
from sklearn import preprocessing
import pandas as pd
import sklearn.metrics as metrics
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

MAX_SEQUENCE_LENGTH = 20
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300


def getTrData(dataFile):
    """
    Prepare the data
    """  
    train = pd.read_csv(dataFile, header=0, delimiter="," )
    R,C=train.shape
    print(C)
    ids=train.iloc[:, 0]
    ids=ids.values.tolist()
    texts=train.iloc[:, 1:(C-1)]
    texts=texts.values.tolist()
    txtData=[]
    for text in texts:
        sent=""
        for s in text:
            s=s.replace("\"","")
            s=s.replace("\'","")
            sent=sent+" "+s
        txtData.append(sent)

    texts=txtData

    
    yL=train.iloc[:, C-1]    
    le = preprocessing.LabelEncoder()
    yL=le.fit_transform(yL)
    labels=list(le.classes_)
    
    label=yL.tolist()
    yC=len(set(label))
    yR=len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y=np.array(y,dtype=np.int32)
    
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    #labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    #print('Shape of label tensor:', labels.shape)    
    #return data,labels,word_index,dim;        
    return data,y,le,labels,ids,word_index,tokenizer

def getDevData(dataFile,tokenizer):
    """
    Prepare the data
    """  
    train = pd.read_csv(dataFile, header=0, delimiter="," )
    R,C=train.shape
    ids=train.iloc[:, 0]
    ids=ids.values.tolist()
    texts=train.iloc[:, 1:C-1]
    texts=texts.values.tolist()
    txtData=[]
    for text in texts:
        sent=""
        for s in text:
            s=s.replace("\"","")
            s=s.replace("\'","")
            sent=sent+" "+s
        txtData.append(sent)

    texts=txtData


    yL=train.iloc[:, C-1]    
    le = preprocessing.LabelEncoder()
    yL=le.fit_transform(yL)
    labels=list(le.classes_)
    
    label=yL.tolist()
    yC=len(set(label))
    yR=len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y=np.array(y,dtype=np.int32)
    
    # finally, vectorize the text samples into a 2D integer tensor
    #tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    #tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    #labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    #print('Shape of label tensor:', labels.shape)    
    #return data,labels,word_index,dim;        
    return data,y,le,labels,ids,word_index    

def loadEmbedding(modelName):
    print('Indexing word vectors.')    
    embeddings_index = {}    
    #f = open(fileName)
    num_features=300    
    #words = np.zeros((num_features,),dtype="float32")
    model = Word2Vec.load(modelName) #gensim trained model
    #model = Word2Vec.load_word2vec_format(modelName, binary=True)    ## google binary model
 
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()    
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index;

def get_WVec(vocabulary,modelName):
    #words = []
    num_features=300    
    #words = np.zeros((num_features,),dtype="float32")
    model = Word2Vec.load(modelName) #gensim trained model
    #model = Word2Vec.load_word2vec_format(modelName, binary=True)    ## google binary model
    
    vocab_size = len(vocabulary)
    #word_idx_map = dict()
    W = np.zeros(shape=(vocab_size, num_features), dtype='float32')            
    #W[0] = np.zeros(num_features, dtype='float32')
    i =0
    np.random.seed(seed=42)
    for word, count in vocabulary.iteritems(): 
        try:
            #words=np.r_[ words, model[word][0:num_features]] 
            W[i]=model[word][0:num_features]
            #word_idx_map[word] = index
            i += 1
        except KeyError:
            rng = np.random.RandomState(123499)        	
            wordVal = rng.randn(num_features)#np.random.random(num_features)
            #word_idx_map[word] = index
            #words=np.r_[ words, word]
            W[i]=wordVal
            i += 1
            continue
    return W
    
def prepareEmbedding(word_index,modelFile):
    
    #num_features=300    
    np.random.seed(seed=42)    
    #words = np.zeros((num_features,),dtype="float32")
    #model = Word2Vec.load(modelName) #gensim trained model    
    model = KeyedVectors.load_word2vec_format(modelFile, binary=False)
    # prepare embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index)+1)
    
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    print(len(embedding_matrix))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        try:
            embedding_vector = model[word][0:EMBEDDING_DIM] #embeddings_index.get(word)
            embedding_matrix[i] = embedding_vector 
        except KeyError:
            try:
                rng = np.random.RandomState()        	
                embedding_vector = rng.randn(EMBEDDING_DIM)#np.random.random(num_features)
                embedding_matrix[i] = embedding_vector
                #print("Not found. "+word)
            except KeyError:    
                continue            
            
        #if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            #embedding_matrix[i] = embedding_vector 
        
    return embedding_matrix;
    
"""
Design model by combining three feature set: acoustic, lexical and psycholinguistic
"""
if __name__ == "__main__":
    
    file_tr_lex = "./data/lex/A4E_emp_train_trans_hseg_2cl_aligned.csv";
    train_x,train_y,le,labels,ids,word_index,tokenizer=getTrData(file_tr_lex)
    
    file_dev_lex = "./data/lex/A4E_emp_dev_trans_autoseg_2cl_aligned.csv";
    dev_x,dev_y,Dle,Dlabels,DIds,_=getDevData(file_dev_lex,tokenizer)
        
    file_tst_lex = "./data/lex/A4E_emp_test_trans_autoseg_2cl_aligned.csv";
    test_x,test_y,Tle,Tlabels,TIds,_=getDevData(file_tst_lex,tokenizer)
    nb_classes = 2
    
    emb_file="/home/shammur/OV/mood/itwac_sensei_paisa_itwiki_a4e.w2v.gensim.bin"
    #embeddings_index=loadEmbedding(emb_file)
    embedding_matrix=prepareEmbedding(word_index,emb_file)
    
    nb_words = min(MAX_NB_WORDS, len(word_index))

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    
    print('Training model.')
    batch_size = 32
    nb_epoch = 12
    nb_classes = 2    
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(nb_classes, activation='softmax')(x)
    
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])
    
    # happy learning!
    model.fit(train_x, train_y, validation_data=(dev_x, dev_y),nb_epoch=nb_epoch, batch_size=batch_size)
    score = model.evaluate([dev_x], dev_y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    score = model.evaluate([dev_x,], dev_y, verbose=0)    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    score = model.evaluate([test_x], test_y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    dev_pred=model.predict_classes([dev_x], batch_size=32, verbose=1)
    test_pred=model.predict_classes([test_x], batch_size=32, verbose=1)
    
    
    y_true=np.argmax(dev_y, axis = 1)
    y_true=Dle.inverse_transform(y_true)
    lab=list(set(y_true.tolist()))
    lab.sort()    
    y_pred=Dle.inverse_transform(dev_pred)
    acc=metrics.accuracy_score(y_true,y_pred)   
    print (acc)
    report=metrics.classification_report(y_true, y_pred)
    print (report)

    #file_dev_ac
    outfileName="results_baseline/lexical_prediction_dev.txt"
    fopen = open(outfileName, "w");
    fopen.write("##InstID\tRef.\tPrediction\tConf\n")    
    for id,ref,pred in zip(DIds,y_true,y_pred):
        fopen.write(str(id)+"\t"+str(ref)+"\t"+str(pred)+"1.0\n")
    fopen.close    
