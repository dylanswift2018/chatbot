#creating a GUI to make it smooth and user friendly
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np 
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents =json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

#functions that perform preprocessing and predicting the text 
def clean_up_sentence(sent) :
    #split words into array 
    sw = nltk.word_tokenize(sent)
    sw = [WordNetLemmatizer.lemmatize(word.lower()) for word in sw]
    return sw

def bow(sen , words, details = True) :
    #tokenize the pattern 
    sw = clean_up_sentence(sen)
    #matrix of words 
    bag = [0] *len(words)
    for x in sw :
        for y,z in enumerate(words):
            if z==x :
                bag[y]=1
                if details :
                    print("found in bag : %s" % z)
    return (np.array(bag))

def predict_class(sen , model) :
    #filter predictions below a threshold
    pr = bow (sen, words , details=False)
    result = model.predict(np.array([pr]))[0]
    ERROR_THRESHOLD = 0.25 
    res = [[x,y] for x,y in enumerate(result) if y > ERROR_THRESHOLD]
    #sorting by probabilty strength 
    res.sort(key= lambda x: x[1] ,reverse = True)
    ret_list =[]
    for x in res :
        ret_list.append({"intent" : classes[x[0]],"probabilty": str(x[1])})
    return ret_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
    
def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res