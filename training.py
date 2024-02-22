import certifi  # import certifi module for SSL certificate handling
import ssl  # import ssl module for SSL-related functionality
import os  # import os module for operating system functionalities
import random  # import random module for generating random numbers
import json  # import json module for JSON-related functionalities
import pickle  # import pickle module for serializing and deserializing Python objects
import numpy as np  # import numpy module for numerical computing
import nltk  # import nltk module for natural language processing tasks
# nltk.download()  # used to download NLTK data modules prior to training
from nltk.stem import WordNetLemmatizer  # import WordNetLemmatizer class for lemmatization
import tensorflow  # import tensorflow module for machine learning tasks
from tensorflow import keras  # import keras module for building neural networks
from keras import Sequential  # import Sequential class for creating sequential neural networks
from keras.layers import Dense, Activation, Dropout  # import layers for constructing neural network architecture
from keras import optimizers  # import optimizers for optimizing neural network parameters
from keras.optimizers import SGD  # import SGD optimizer for stochastic gradient descent

# initialize lemmatizer for NLP processing and intents.json file used to store chatbot knowledge base
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# lists used to store processed words, required classes, combination documents, and ignored characters to not be processed
words = []
classes = []
documents = []
ignore_letters = ['?', ',', '!', '.']

# iterate through intents.json to access patterns and responses for user input
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        documents.append((word_list, intent['tag']))
        words.extend(word_list)  # Append words to the 'words' list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# iterate words list and initialize lemmatizer to break down to its NLP root meaning -- if the word is not contained in ignored_characters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# store populated lists as writing binaries using pickle 
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# initialize an empty list to store training data and create an output list filled with zeros
training = []
output_empty = [0] * len(classes)

# format data to numerical formatting for machine learning training and store in initialized training list
for document in documents:
    bag = []  # initialize an empty list to store bag of words
    word_patterns = document[0] # get the word list from the document tuple
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # create a bag of words with 1s and 0s indicating word presence
    bag = [1 if word in word_patterns else 0 for word in words]
    output_row = list(output_empty) # copy the output_empty list
    output_row[classes.index(document[1])] = 1 # set the corresponding class index to 1

    # append the bag and output_row as separate lists to the training list
    training.append([bag, output_row])

# shuffle the training data
random.shuffle(training)

# split the training data into input (X) and output (Y)
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

# initialize machine learning model and set required layers parameters
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) # input layer
model.add(Dropout(0.5)) # dropout layer to prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) # output layer
# initialize Stochastic Gradient Boosting optimizer
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# train the model and set to histogram variable for model.save process
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model_neo.h5', hist) # save the trained model 
print("Done.") # output "Done." when training is complete for verification