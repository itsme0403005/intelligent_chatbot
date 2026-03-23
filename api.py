import json
import random
import numpy as np
import tensorflow as tf
import pickle
import re
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

CONFIDENCE_THRESHOLD = 0.6

with open("intent.json") as file:
    intents = json.load(file)

model = tf.keras.models.load_model("chatbot_model.keras")

tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))
MAX_LEN = pickle.load(open("max_len.pkl", "rb"))

conversation_state = None

@app.route("/")
def home():
    return "Chatbot API running 🚀"

def extract_order_id(text):
    match = re.search(r"\d+", text)
    if match:
        return match.group()
    return None

def predict_intent(text):

    seq = tokenizer.texts_to_sequences([text.lower()])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    prediction = model.predict(padded, verbose=0)

    confidence = np.max(prediction)
    index = np.argmax(prediction)

    tag = encoder.inverse_transform([index])[0]

    return tag, confidence

def get_response(tag):

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry I didn't understand."

@app.route("/chat", methods=["GET", "POST"])
def chat():

    global conversation_state

    if request.method == "GET":
        return jsonify({
            "message": "Chatbot API is working. Send a POST request with JSON: { 'message': 'your text' }"
        })

    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"response": "Invalid request"}), 400

    message = data["message"]

    order_id = extract_order_id(message)


    if conversation_state == "waiting_order_id":

        if order_id:
            conversation_state = None
            return jsonify({
                "response": f"Order {order_id} is being processed and will arrive soon 🚚"
            })
        else:
            return jsonify({"response": "Please provide a valid order ID."})

    tag, confidence = predict_intent(message)

    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({
            "response": "I'm not sure I understood. Could you rephrase?"
        })

    if tag == "order_status":

        if order_id:
            return jsonify({
                "response": f"Order {order_id} is currently being processed and will arrive in 3 days 🚚"
            })

        conversation_state = "waiting_order_id"
        return jsonify({"response": "Please provide your order ID."})

    response = get_response(tag)

    return jsonify({"response": response})

import os

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)