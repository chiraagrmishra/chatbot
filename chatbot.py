from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load pre-trained model and other necessary files
model = tf.keras.models.load_model('chatbot_model.h5')
lemmatizer = WordNetLemmatizer()

with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

with open('intents.json') as f:
    intents = json.load(f)

def preprocess_message(message):
    # Tokenize the message
    word_list = nltk.word_tokenize(message)
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]
    
    # Create bag of words array
    bag = [0] * len(words)
    for w in word_list:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data['message']
    
    # Preprocess the message
    preprocessed_message = preprocess_message(message)
    
    # Predict the intent
    prediction = model.predict(np.array([preprocessed_message]))[0]
    intent_index = np.argmax(prediction)
    intent = classes[intent_index]
    
    # Create a list of intent probabilities
    intent_list = [{"intent": intent, "probability": str(prediction[intent_index])}]
    
    # Get the response from the intents file
    response = get_response(intent_list, intents)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)