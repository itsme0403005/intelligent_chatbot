import json
import random
import numpy as np
import tensorflow as tf
import pickle
import re
from difflib import get_close_matches
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load files
# -----------------------------

with open("intent.json") as file:
    intents = json.load(file)

model = tf.keras.models.load_model("chatbot_model.keras")

tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))
MAX_LEN = pickle.load(open("max_len.pkl", "rb"))

CONFIDENCE_THRESHOLD = 0.6
conversation_state = None


# -----------------------------
# Clean text (same as training)
# -----------------------------

def clean_text(text):

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.strip()

    return text


# -----------------------------
# Spelling correction
# -----------------------------

corrections = {
    "hii": "hi",
    "helloo": "hello",
    "oder": "order",
    "ordr": "order",
    "cncle": "cancel",
    "cancle": "cancel",
    "statuus": "status"
}


def correct_spelling(text):

    words = text.split()
    vocab = list(tokenizer.word_index.keys())

    corrected = []

    for word in words:

        if word in corrections:
            corrected.append(corrections[word])
            continue

        match = get_close_matches(word, vocab, n=1, cutoff=0.75)

        if match:
            corrected.append(match[0])
        else:
            corrected.append(word)

    return " ".join(corrected)


# -----------------------------
# Extract order ID
# -----------------------------

def extract_order_id(text):

    match = re.search(r"\d+", text)

    if match:
        return match.group()

    return None


# -----------------------------
# Predict intent
# -----------------------------

def predict_intent(text):

    text = clean_text(text)
    text = correct_spelling(text)

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    prediction = model.predict(padded, verbose=0)

    confidence = np.max(prediction)
    index = np.argmax(prediction)

    tag = encoder.inverse_transform([index])[0]

    return tag, confidence


# -----------------------------
# Get response
# -----------------------------

def get_response(tag):

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand."


# -----------------------------
# Chat loop
# -----------------------------

print("Chatbot running... (type quit to exit)")

while True:

    user_input = input("You: ").strip()

    if user_input.lower() == "quit":
        print("Bot: Goodbye! Have a nice day.")
        break


    order_id = extract_order_id(user_input)


    # -----------------------------
    # HANDLE CONTEXT FIRST
    # -----------------------------

    if conversation_state:

        if order_id:

            if conversation_state == "order_status":
                print(f"Bot: Order {order_id} is currently being processed and will arrive in 3 days 🚚")

            elif conversation_state == "cancel_order":
                print(f"Bot: Order {order_id} has been successfully cancelled.")

            elif conversation_state == "refund_request":
                print(f"Bot: Refund for order {order_id} has been initiated and will reflect within 5–7 days.")

            conversation_state = None
            continue

        else:
            print("Bot: Please provide a valid order ID.")
            continue


    # -----------------------------
    # If message contains order ID
    # -----------------------------

    if order_id and not conversation_state:

        # Try predicting intent from text
        tag, confidence = predict_intent(user_input)

        if tag == "order_status":
            print(f"Bot: Order {order_id} is currently being processed and will arrive in 3 days 🚚")

        elif tag == "cancel_order":
            print(f"Bot: Order {order_id} has been successfully cancelled.")

        elif tag == "refund_request":
            print(f"Bot: Refund for order {order_id} has been initiated and will reflect within 5–7 days.")

        else:
            print("Bot: Please tell me what you want to do with this order (track, cancel, refund).")

        continue


    # -----------------------------
    # Predict intent
    # -----------------------------

    tag, confidence = predict_intent(user_input)

    if confidence < CONFIDENCE_THRESHOLD:
        print("Bot: I'm not sure I understood. Could you rephrase?")
        continue


    # -----------------------------
    # Intent logic
    # -----------------------------

    if tag == "order_status":

        print("Bot: Kindly share your order number and I will check it for you.")
        conversation_state = "order_status"


    elif tag == "cancel_order":

        print("Bot: Please provide your order ID to cancel the order.")
        conversation_state = "cancel_order"


    elif tag == "refund_request":

        print("Bot: Please provide your order ID for refund processing.")
        conversation_state = "refund_request"


    else:

        response = get_response(tag)
        print("Bot:", response)