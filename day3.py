import joblib
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Load final cleaned dataset
df = pd.read_csv("data/final_cleaned_spam.csv")

print("----- DATASET LOADED -----")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

# Separate input and target

X = df["clean_message"]
y = df["label"]

print("\n----- X AND Y -----")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst 5 X values:")
print(X.head())

print("\nFirst 5 y values:")
print(y.head())

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n----- TRAIN TEST SPLIT -----")
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

print("\n----- TRAINING LABEL DISTRIBUTION -----")
print(y_train.value_counts())

print("\n----- TESTING LABEL DISTRIBUTION -----")
print(y_test.value_counts())

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Learn vocabulary from training data and transform it
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform testing data using the same vocabulary
X_test_tfidf = vectorizer.transform(X_test)

print("\n----- TF-IDF -----")
print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

# Create Naive Bayes model
model = MultinomialNB()
# ----- COMPLEMENT NAIVE BAYES -----

model_cnb = ComplementNB()

model_cnb.fit(X_train_tfidf, y_train)

# Save the trained model and TF-IDF vectorizer
joblib.dump(model_cnb, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\n----- MODEL SAVED -----")
print("Spam model saved successfully!")
print("TF-IDF vectorizer saved successfully!")

y_pred_cnb = model_cnb.predict(X_test_tfidf)

print("\n----- COMPLEMENT NAIVE BAYES -----")
print(classification_report(y_test, y_pred_cnb))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_cnb))

# Train the model
model.fit(X_train_tfidf, y_train)

print("\n----- MODEL TRAINING -----")
print("Model trained successfully!")

# Make predictions on test data
y_pred = model.predict(X_test_tfidf)

print("\n----- PREDICTIONS -----")
print("First 10 actual labels:")
print(y_test.head(10).values)

print("\nFirst 10 predicted labels:")
print(y_pred[:10])

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n----- MODEL ACCURACY -----")
print("Accuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")

# Detailed model evaluation
print("\n----- CLASSIFICATION REPORT -----")
print(classification_report(y_test, y_pred))

print("\n----- CONFUSION MATRIX -----")
print(confusion_matrix(y_test, y_pred))

# Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["ham", "spam"]
)

disp.plot()
plt.title("Spam Detection Confusion Matrix")
plt.show()

# ----- TEST A NEW MESSAGE -----

new_message = ["Congratulations! You have won a free prize. Claim now!"]

# Clean the message
cleaned_message = [clean_text(new_message[0])]

# Convert message into TF-IDF
new_message_tfidf = vectorizer.transform(cleaned_message)

# Predict
prediction = model_cnb.predict(new_message_tfidf)

print("\n----- NEW MESSAGE TEST -----")
print("Message:", new_message[0])
print("Prediction:", prediction[0])

# Test a normal message
new_message = ["Hey bro, are we meeting tomorrow?"]

cleaned_message = [clean_text(new_message[0])]

new_message_tfidf = vectorizer.transform(cleaned_message)

prediction = model_cnb.predict(new_message_tfidf)

print("\n----- NORMAL MESSAGE TEST -----")
print("Message:", new_message[0])
print("Prediction:", prediction[0])

# ----- INTERACTIVE SPAM DETECTOR -----

print("\n===================================")
print("        SPAM DETECTOR")
print("===================================")
print("Type 'exit' to close the detector.")

while True:
    user_message = input("\nEnter a message: ")

    if user_message.lower() == "exit":
        print("\nExiting spam detector...")
        break

    cleaned_message = clean_text(user_message)

    message_tfidf = vectorizer.transform([cleaned_message])

    prediction = model_cnb.predict(message_tfidf)

    print("-----------------------------------")

    if prediction[0] == "spam":
        print("🚨 Result: SPAM")
    else:
        print("✅ Result: HAM")

    print("-----------------------------------")