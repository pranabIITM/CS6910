import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import pathlib
import keras
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.layers import SimpleRNN,LSTM,GRU,Embedding,Dense,Dropout,Input
from tensorflow.keras.optimizers import Adam,Nadam
from keras import Model

def tok_map(data):
    #Taking the columns from the data set provided
    source = data['en'].values
    target = data['hi'].values
    target = '\t'+target+'\n'

    len_list_s = [len(i) for i in source]
    s_max_len = max(len_list_s)

    len_list_t = [len(i) for i in target]
    t_max_len = max(len_list_t)

    # Creating token set and token mapping for source language
    s_tok = set()
    for sw in source:
        for chr in sw:
            s_tok.add(chr)
    source_tokens = sorted(list(s_tok))
    s_tok_map = dict([(chr,i+1) for i,chr in enumerate(source_tokens)])
    s_tok_map[" "] = 0

    # Creating token set and token mapping for Target language
    t_tok = set()
    for st in target:
        for chr in st:
            t_tok.add(chr)
    tar_tokens = sorted(list(t_tok))
    t_tok_map = dict([(chr,i+1) for i,chr in enumerate(tar_tokens)])
    t_tok_map[" "] = 0

    return source_tokens, s_tok_map, s_max_len, tar_tokens, t_tok_map, t_max_len

def dataLoad(path):
    with open(path) as dataFile:
        dataset = pd.read_csv(dataFile,sep='\t',header=None,names=["hi","en",""],skip_blank_lines=True,index_col=None)
    #print(dataset.head())
    dataset = dataset[dataset['hi'].notna()]
    #print(dataset.head())
    dataset = dataset[dataset['en'].notna()]
    #print(dataset.head())
    dataset = dataset[['hi','en']]
    #print(dataset.head())
    return dataset

def dataProcess(data):
    src,tar = data['en'].values, data['hi'].values
    tar = "\t" + tar + "\n"

    slen = len(src)
    enc_inp = np.zeros(
        (slen,s_max_len), dtype="float32"
    )

    tlen = len(tar)
    dec_inp = np.zeros(
        (tlen,t_max_len), dtype="float32"
    )
    dec_tar = np.zeros(
        (tlen, t_max_len, len(tar_tokens)+1), dtype="int"
    )
    for i,(sw,tw) in enumerate(zip(src,tar)):
        #enmurating the source data and creating input data set
        for j,ch in enumerate(sw):
            enc_inp[i,j] = s_tok_map[ch]
        enc_inp[i,j+1:] = s_tok_map[" "]

        #enmurating the Target data and creating decoder input data set target output
        for j,ch in enumerate(tw):
            dec_inp[i,j] = t_tok_map[ch]
            if j>0:
                dec_tar[i,j-1,t_tok_map[ch]] = 1
        dec_inp[i,j+1:] = t_tok_map[" "]
        dec_tar[i,j:,t_tok_map[" "]] = 1
        
    return enc_inp, dec_inp, dec_tar

from sys import argv

if(len(argv) != 11):
    print("Invalid num of parameters passed ")
    exit()

#Parsing the command line argumants
Lay = argv[1]
nu = int(argv[2])
enclay = int(argv[3])
declay = int(argv[4])
embd = int(argv[5])
d_n = int(argv[6])
d = float(argv[7])
epo = int(argv[8])
BATCH_SIZE = int(argv[9])
ler = float(argv[10])

print("Provided Parameters are: ")
print(f'Cell = {Lay}')
print(f'Units = {nu}')
print(f'Enc Layer = {enclay}')
print(f'Dec Layer = {declay}')
print(f'Embed size = {embd}')
print(f'Dense size = {d_n}')
print(f'Dropout = {d}')
print(f'Epochs = {epo}')
print(f'Batch size = {BATCH_SIZE}')
print(f'LR = {ler}')

#Loading the dataset from the dakshina_dataset_v1.0
train = dataLoad("/content/drive/MyDrive/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv")
source_tokens, s_tok_map, s_max_len, tar_tokens, t_tok_map, t_max_len = tok_map(train)
dev = dataLoad("/content/drive/MyDrive/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv")
test = dataLoad("/content/drive/MyDrive/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv")

# Process the training data
train_encoder_input, train_decoder_input, train_decoder_target = dataProcess(train)

# Process the validation data
val_encoder_input, val_decoder_input, val_decoder_target = dataProcess(dev)

def seq2seqModel(Layer = "LSTM", nunits = 32, encl = 2, decl = 2,embds = 32,dense_size=32,dropout=None):
    keras.backend.clear_session()
    # source_tokens, s_tok_map, s_max_len, tar_tokens, t_tok_map, t_max_len
    enc_inps = Input(shape=(None,))
    enc_emb = Embedding(input_dim=len(source_tokens)+1, output_dim = embds, mask_zero=True)
    encop = enc_emb(enc_inps)

    dec_inps = Input(shape=(None,))
    dec_emb = Embedding(input_dim = len(tar_tokens)+1,output_dim = embds,mask_zero=True)

    # If the cell type is chosen as RNN ----------------------------------------------------
    if Layer == "RNN":
        encLays = [SimpleRNN(nunits,return_sequences=True) for i in range(encl-1)]
        encLast = SimpleRNN(nunits,return_state=True)
        encmb = encop
        for enLay in encLays:
            encmb = enLay(encmb)
            if dropout is not None:
                encmb = Dropout(dropout)(encmb)

        _, state = encLast(encmb)
        encoder_states = state
        
        decoder = [SimpleRNN(nunits,return_sequences=True,return_state=True) for i in range(decl)]
        decEmbop = dec_emb(dec_inps)
        dLhInp,_ = decoder[0](decEmbop,initial_state=state)
        for i in range(1,decl):
            dLhInp,_ = decoder[i](dLhInp,initial_state=state)

    # If the cell type is chosen as LSTM ----------------------------------------------------    
    elif Layer == "LSTM":
        encLays = [LSTM(nunits,return_sequences=True) for i in range(encl-1)]
        encLast = LSTM(nunits,return_state=True)
        encmb = encop
        for enLay in encLays:
            encmb = enLay(encmb)
            if dropout is not None:
                encmb = Dropout(dropout)(encmb)
            
        _, state_h,state_c = encLast(encmb)
        encoder_states = [state_h,state_c]
        
        decoder = [LSTM(nunits,return_sequences=True,return_state=True) for i in range(decl)]
        
        decEmbop = dec_emb(dec_inps)
        dLhInp,_,_ = decoder[0](decEmbop,initial_state=encoder_states)
        for i in range(1,decl):
            dLhInp,_,_ = decoder[i](dLhInp,initial_state=encoder_states)

    # If the cell type is chosen as GRU ----------------------------------------------------  
    elif Layer == "GRU":
        encLays = [GRU(nunits,return_sequences=True) for i in range(encl-1)]
        encLast = GRU(nunits,return_state=True)
        encmb = encop
        for enLay in encLays:
            encmb = enLay(encmb)
            if dropout is not None:
                encmb = Dropout(dropout)(encmb)
            
        _, state = encLast(encmb)
        encoder_states = state
        
        decoder = [GRU(nunits,return_sequences=True,return_state=True) for i in range(decl)]
        
        decEmbop = dec_emb(dec_inps)
        dLhInp,_ = decoder[0](decEmbop,initial_state=state)
        for i in range(1,decl):
            dLhInp,_ = decoder[i](dLhInp,initial_state=state)
            
        
    DLayerH = Dense(dense_size, activation='relu')
    preact = DLayerH(dLhInp)
    DL_O = Dense(len(tar_tokens)+1, activation = 'softmax')
    act_op = DL_O(preact)
    
    train_model = Model([enc_inps,dec_inps],act_op)

    return train_model

    
#Layer = "LSTM", nunits = 32, encl = 2, decl = 2,embds = 32,dense_size=32,dropout=None


train = seq2seqModel(Layer = Lay, nunits=nu, encl=enclay, decl=declay, embds = embd, dense_size=d_n, dropout=d)
train.compile(optimizer = Adam(learning_rate=ler),loss='categorical_crossentropy',metrics=['accuracy'])
train.fit([train_encoder_input,train_decoder_input],train_decoder_target, batch_size=BATCH_SIZE,
           validation_data = ([val_encoder_input,val_decoder_input],val_decoder_target),
           epochs=epo)

"""
    README Q1 --------------------------------------------------------------------
        
        To compile the file with command line arguments write in following format in terminal :-
        $ python filename.py LayerType NoOfRNN_Units encoderLayers decoderLayers embeddingSize denseSize dropout Epochs BatchSize LearningRate
        
        Example:-
        $ python AS3_Q1.py LSTM 32 2 2 16 32 0.2 2 64 0.001
    
"""
