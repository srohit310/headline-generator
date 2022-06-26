import numpy as np
import pandas as pd
import pickle
from statistics import mode
import nltk
from nltk import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input,LSTM,Embedding,Dense,Concatenate,Attention,Bidirectional
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import re
from os import system

def clean(texts,src):
    #remove the html tags
    texts = BeautifulSoup(texts, "lxml").text
    #tokenize the text into words 
    words=word_tokenize(texts.lower())
    #filter words which contains \ 
    #integers or their length is less than or equal to 3
    words= list(filter(lambda w:(w.isalpha() and len(w)>=3),words))
    #contraction file to expand shortened words
    words= [contractions[w] if w in contractions else w for w in words ]
    #remove special characters
    words = [re.sub('[^A-Za-z0-9]+', '', x) for x in words]
    #stem the words to their root word and filter stop words
    if src=="inputs":
        words= [stemm.stem(w) for w in words if w not in stop_words]
    else:
        words= [w for w in words if w not in stop_words]
    return words

class Inference:

    def init_var(self, in_tokenizer, tr_tokenizer, max_in_len, max_tr_len):

        self.model = models.load_model("newsreport_s2s")
        
        self.latent_dim = 500
        self.max_in_len = max_in_len
        self.max_tr_len = max_tr_len

        self.reverse_target_word_index = tr_tokenizer.index_word
        self.reverse_source_word_index = in_tokenizer.index_word
        self.target_word_index = tr_tokenizer.word_index
        self.reverse_target_word_index[0]=' ' 
        
    def building_en_inf(self):

        en_outputs,state_enc, cell_enc = self.model.layers[6].output
        en_states=[state_enc, cell_enc]
        return Model(self.model.input[0],[en_outputs]+en_states)

    def building_dec_inf(self):

        dec_state_h = Input(shape=(self.latent_dim,))
        dec_state_c = Input(shape=(self.latent_dim,))
        dec_hidden_state_input = Input(shape=(self.max_in_len,self.latent_dim))

        dec_inputs = Input(shape=(None,)) 
        dec_emb_layer = self.model.layers[5]
        dec_lstm = self.model.layers[7]
        dec_embedding= dec_emb_layer(dec_inputs)

        dec_outputs2, dec_h2, dec_c2 = dec_lstm(dec_embedding, initial_state=[dec_state_h,dec_state_c])

        attention = self.model.layers[8]
        attn_out2 = attention([dec_outputs2,dec_hidden_state_input])

        merge2 = Concatenate(axis=-1)([dec_outputs2, attn_out2])

        dec_dense = self.model.layers[10]
        dec_outputs2 = dec_dense(merge2)

        return Model(
        [dec_inputs] + [dec_hidden_state_input,dec_state_h,dec_state_c],
        [dec_outputs2] + [dec_h2, dec_c2])
    
    def __init__(self, in_tokenizer, tr_tokenizer, max_in_len, max_tr_len):

        self.init_var(in_tokenizer, tr_tokenizer, max_in_len, max_tr_len)
        self.en_model = self.building_en_inf()
        self.dec_model = self.building_dec_inf()

    def decode_sequence(self, input_seq):

        en_out, en_h, en_c = self.en_model.predict(input_seq)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.target_word_index['sos']
        stop_condition = False
        decoded_sentence = ""
        print(target_seq)

        while not stop_condition: 

            output_words, dec_h, dec_c = self.dec_model.predict([target_seq] + [en_out,en_h, en_c])
            word_index = np.argmax(output_words[0, -1, :])
            text_word = self.reverse_target_word_index[word_index]
            decoded_sentence += text_word +" "
            if text_word == "eos" or len(decoded_sentence) > self.max_tr_len:
              stop_condition = True
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = word_index
            en_h, en_c = dec_h, dec_c

        return decoded_sentence

if __name__ == '__main__':

    contractions=pickle.load(open("contractions.pkl","rb"))['contractions']
    stop_words=set(stopwords.words('english'))
    stemm=LancasterStemmer()
    max_in_len = 190
    max_tr_len = 54

    with open('in_tokenizer.pickle', 'rb') as handle:
        in_tokenizer = pickle.load(handle)

    with open('tr_tokenizer.pickle', 'rb') as handle:
        tr_tokenizer = pickle.load(handle)

    inference = Inference(in_tokenizer,tr_tokenizer,max_in_len,max_tr_len)
    system('cls')
    inp_review = input("Enter : ")
    print("Review :",inp_review)
    inp_review = clean(inp_review,"inputs")
    inp_review = ' '.join(inp_review)
    inp_x = in_tokenizer.texts_to_sequences([inp_review]) 
    inp_x = pad_sequences(inp_x,  maxlen=max_in_len, padding='post')
    
    summary = inference.decode_sequence(inp_x.reshape(1,max_in_len))
    if 'eos' in summary :
      summary=summary.replace('eos','')
    print("\nPredicted summary:",summary);print("\n")