import random
import json
import pickle
import numpy as np

import nltk  
nltk.download('punkt')  
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []

output_empty = [0]*len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
    
random.shuffle(training)

train_x = []
train_y = []

for (bag, output_row) in training:
    train_x.append(bag)
    train_y.append(output_row)

train_x = np.array(train_x)
train_y = np.array(train_y)

class ChatBotModel(nn.Module):
    def __init__(self):
        super(ChatBotModel, self).__init__()
        self.fc1 = nn.Linear(len(train_x[0]), 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, len(train_y[0]))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = self.fc3(x) 
        return x

model = ChatBotModel()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_function = nn.CrossEntropyLoss()

# Convert lists to tensors
train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y).long()

for epoch in range(5000):
    optimizer.zero_grad()
    output = model(train_x)
    loss = loss_function(output, torch.argmax(train_y, dim=1)) # Adjusted here
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    _, predicted = torch.max(output, 1)
    correct = (predicted == torch.argmax(train_y, dim=1)).sum().item() # Adjusted here
    total = train_y.size(0)
    accuracy = correct / total

    print(f'Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy}')

torch.save(model.state_dict(), 'chatbot_model.pth')
print('Done')
