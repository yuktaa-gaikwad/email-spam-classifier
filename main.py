import pandas as pd
import re
import string
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# Load dataset
file_path = "spam.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("Dataset 'spam.csv' not found. Please ensure the file is in the correct directory.")

df = pd.read_csv(file_path, encoding='latin-1')

# Keep only relevant columns and rename them
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

df['text'] = df['text'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_predictions = nb_model.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
svm_predictions = svm_model.predict(X_test_tfidf)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# Save the best model
best_model = nb_model if nb_accuracy > svm_accuracy else svm_model
with open("vectorizer.pkl", "wb") as vec_file, open("classifier.pkl", "wb") as clf_file:
    pickle.dump(vectorizer, vec_file)
    pickle.dump(best_model, clf_file)

def predict_email(email_text):
    if not os.path.exists("vectorizer.pkl") or not os.path.exists("classifier.pkl"):
        raise FileNotFoundError("Model files not found. Train the model first.")
    with open("vectorizer.pkl", "rb") as vec_file, open("classifier.pkl", "rb") as clf_file:
        vectorizer = pickle.load(vec_file)
        classifier = pickle.load(clf_file)
    processed_text = preprocess_text(email_text)
    email_vector = vectorizer.transform([processed_text])
    prediction = classifier.predict(email_vector)
    return "Spam" if prediction[0] == 1 else "Ham"

if __name__ == "__main__":
    email_text = input("Enter email content: ")
    result = predict_email(email_text)
    print(f"The email is classified as: {result}")
