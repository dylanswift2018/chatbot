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

 # lemmatizing lower each word and remove duplicates 
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

words = sorted(list(set(words)))

# sort classes 
classes = sorted(list(set(classes)))

#documents = combination between patterns and intents 
print(len(documents), "documents")

# classes = intents 
print(len(words) , "unique lemmatized words" , words)

pickle.dump(words, open('words.pkl' , 'wb'))
pickle.dump(classes , open('classes.pkl','wb'))

# creating training and testing data 
train = []
#creating empty array for output 
output_empty = [0] *len(classes)
#training set we have a bag of words for each sentence 
for doc in documents :
    #init the bag 
    bag = []
    #list of tokenized words for the pattern 
    pattern_words = doc[0]
    #lemmatizing each word for the pattern 
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    #creating our bag of words array with 1 if the word match found in current pattern 
    for w in words: 
        bag.append(1)  if w in pattern_words else bag.append(0)

    #output is 0 for each tag and 1 for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    train.append([bag , output_row])

#shuffling the features and turining it into an np.array 
random.shuffle(train)
train = np.array(train)
#creating training and testin lists X patterns and Y intents 
train_x = list(train[:,0])
train_y = list(train[:,1])
print("Training data created")

# building the model 
#we gonna build a DNN that has 3 layers usin keras sequential API

model = Sequential()
model.add(Dense(128 , input_shape = (len(train_x[0]),),activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation ='softmax'))

#compiling the model 
sgd = SGD(lr =0.01, decay=1e-6, momentum=0.9,nesterov = True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x),np.array(train_y),epochs=200, batch_size=5,verbose=1)
model.save('chatbot_model.h5',hist)

print("model created :) ")