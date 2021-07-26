# we import necessary packages and modules 
import nltk 
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle

import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Activation ,Dropout
from keras.optimizers import SGD 
import random 

#initializing the parameters we're going to use 
words = []
classes = []
documents = []
ignore_words = ['?' , '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

#preprocessing data 
#we should tokenize the words and create a list of classes for our tags 
for intent in intents['intents'] :
    for pat in intent['patterns'] :
        #we tokenize each word
        w = nltk.word_tokenize(pat)
        words.extend(w)
        
        #add docs in the corpus
        documents.append((w , intent['tag']))

        #add to our classes list 
        if intent['tag'] not in classes :
            classes.append(intent['tag'])
            

    