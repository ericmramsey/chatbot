import random  # import random module for generating random numbers
import json  # import json module for JSON-related functionalities
import pickle  # import pickle module for serializing and deserializing Python objects
import numpy as np  # import numpy module for numerical computing
import nltk  # import nltk module for natural language processing tasks
from nltk.stem import WordNetLemmatizer  # import WordNetLemmatizer class for lemmatization
from keras.models import load_model  # import load_model function to load pre-trained model


# initialize lemmatizer for lemmatization and load intens.json knowledge base
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# load words/classes from pickle files and initialize machine learning model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model_neo.h5')

# function used to clean and tokenize user input for further processing
def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence) # tokenize user input into individual words
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# function to create bag of words representation for input sentence
def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence) # clean and tokenize input sentence
    bag = [0] * len(words) # initialize bag of words and numerically format with zeros to start
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1 # set corresponding index to 1 if word is present in vocabulary
    return np.array(bag)

# function used to predict intent based on what is contained in user input
def predict_intents(sentence):
    bow = bag_of_words(sentence) # initialize new bag of words representation for input sentence
    res = model.predict(np.array([bow]))[0] # make prediction using pre-trained model
    ERROR_THRESHOLD = 0.25 # define error threshold for model predictions
    
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # filter predictions above error threshold and store in return_list
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    # format results as intent and probability pairs
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# function used to get response based on predicted intents
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent'] # obtain predicted intent
    list_of_intents = intents_json['intents'] # obtain lists of intents from intents.json knowledge base
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses']) # determine random response from the matching intent
            break
    return result

# 
while True:
    # obtain user input for further processing
    user_input = input("You: ") 
    
    # end chatbot session if user_input contains "exit"
    if user_input.lower() == "exit":
        break
    
    ints = predict_intents(user_input) # call predict_intents function to determine 
    res = get_response(ints, intents)
    print("Chatbot: ", res)