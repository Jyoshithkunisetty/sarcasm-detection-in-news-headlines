from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import webbrowser

app = Flask(__name__)

model = load_model("sarcasm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    headline = request.form['headline']

    seq = tokenizer.texts_to_sequences([headline])
    pad = pad_sequences(seq, maxlen=40)

    prediction = model.predict(pad)

    prob = prediction[0][0]

    print("Prediction Probability:", prob)

    if prob >= 0.7:
        result = "Sarcastic"
    else:
        result = "Not Sarcastic"

    return render_template('index.html', prediction=result, headline=headline)


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
