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


