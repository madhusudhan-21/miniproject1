from flask import Flask, request, jsonify
import joblib
import re

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# Clean the input message
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Home route
@app.route("/")
def home():
    return "Spam Detector Backend is Running!"


# Predict whether a message is spam or ham
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Please provide a valid message."}), 400

    cleaned_message = clean_text(message)
    message_tfidf = vectorizer.transform([cleaned_message])
    prediction = model.predict(message_tfidf)[0]

    return jsonify({
        "message": message,
        "prediction": prediction.upper()
    })


if __name__ == "__main__":
    app.run(debug=True)
