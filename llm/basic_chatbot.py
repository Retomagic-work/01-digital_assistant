import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import torch
import torch.nn as nn
import torch.nn.functional as F

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

class ChatBotModel(nn.Module):  
    def __init__(self):  
        super(ChatBotModel, self).__init__()  
        self.fc1 = nn.Linear(len(words), 256)  
        self.fc2 = nn.Linear(256, 64)  
        self.fc3 = nn.Linear(64, len(classes))  
  
    def forward(self, x):  
        x = F.relu(self.fc1(x))  
        x = F.dropout(x, p=0.5)  
        x = F.relu(self.fc2(x))  
        x = F.dropout(x, p=0.5)  
        x = self.fc3(x)   
        return x  

model = ChatBotModel()  
model.load_state_dict(torch.load('chatbot_model.pth'))  
model.eval()  

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    bow = torch.from_numpy(bow).float()
    res = model(bow)
    prob, predicted = torch.max(F.softmax(res, dim=0), 0)
    if prob.item() < 0.75:  # define your threshold here
        return None
    else:
        return [{'intent': classes[predicted.item()], 'probability': prob.item()}]


def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    else:
        result = random.choice(["Sorry, I didn't understand that. Could you please repeat it?", 
                                "I'm not sure I follow. Could you please rephrase your question?", 
                                "I'm sorry, but I'm unable to assist with that.", 
                                "Can you please say that in a different way?", 
                                "I'm sorry, could you please provide more details?"])
    return result



def predict_class(sentence):
    bow = bag_of_words(sentence)
    bow = torch.from_numpy(bow).float()
    res = model(bow)
    _, predicted = torch.max(res, 0)
    return [{'intent': classes[predicted.item()], 'probability': torch.max(F.softmax(res, dim=0)).item()}]

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result