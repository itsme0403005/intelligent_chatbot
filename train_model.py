import json
import numpy as np
import tensorflow as tf
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import pickle


# -----------------------------
# Clean text
# -----------------------------

def clean_text(text):

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.strip()

    return text


# -----------------------------
# Load dataset
# -----------------------------

with open("intent.json") as file:
    data = json.load(file)

sentences = []
labels = []

for intent in data["intents"]:

    for pattern in intent["patterns"]:

        cleaned = clean_text(pattern)

        sentences.append(cleaned)
        labels.append(intent["tag"])


# -----------------------------
# Encode labels
# -----------------------------

encoder = LabelEncoder()
y = encoder.fit_transform(labels)
y = tf.keras.utils.to_categorical(y)


# -----------------------------
# Tokenization
# -----------------------------

tokenizer = Tokenizer(
    num_words=10000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)


# -----------------------------
# Padding
# -----------------------------

MAX_LEN = max(len(seq) for seq in sequences)

padded = pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    padding="post"
)


# -----------------------------
# Save preprocessing objects
# -----------------------------

pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
pickle.dump(encoder, open("label_encoder.pkl", "wb"))
pickle.dump(MAX_LEN, open("max_len.pkl", "wb"))


# -----------------------------
# Build model
# -----------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(
        input_dim=10000,
        output_dim=128,
        input_length=MAX_LEN
    ),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dense(len(set(labels)), activation="softmax")

])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# Training
# -----------------------------

print("Training model...")

early_stop = EarlyStopping(
    monitor="loss",
    patience=10,
    restore_best_weights=True
)

model.fit(
    padded,
    y,
    epochs=300,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)


# -----------------------------
# Save model
# -----------------------------

model.save("chatbot_model.keras")

print("Training completed!")