import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

# Load dataset
data = []
with open(r"D:\3-2\learnathon\frontend\Sarcasm_Headlines_Dataset.json") as f:

    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

sentences = df['headline']
labels = df['is_sarcastic']

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=40)

X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2
)

model = Sequential()
model.add(Embedding(10000, 64, input_length=40))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

model.save("sarcasm_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model Saved Successfully")
