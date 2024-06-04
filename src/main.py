import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import re

# Load data from csv
def load_data(path):
    data = pd.read_csv(path)
    data.columns = ['label', 'text']
    return data

# Further preprocess data
def preprocess(text):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Lowercase text
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Ecnode labels of a dataset
def encode_labels(data):
    label_enc = LabelEncoder()
    data['label'] = label_enc.fit_transform(data['label'])
    return data, label_enc

# Extract features from datasets
def extract_features(train_text, test_text, max_features = 1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)
    return x_train, x_test, vectorizer

# Train and return a LR model
def train_model(x_train, y_train):
    model = LogisticRegression(class_weight='balanced')
    model.fit(x_train, y_train)
    return model

# Return various statistics on the model
def evaluate_model(model, x, y):
    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    return accuracy, report

def main():
    if len(sys.argv) <= 2:
        print("Usage: python3 main.py [path to training data] [path to test data]")
        return
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    train_data = load_data(train_file)
    test_data = load_data(test_file)
    
    train_data['processed_text'] = train_data['text'].apply(preprocess)
    test_data['processed_text'] = test_data['text'].apply(preprocess)

    train_data, label_enc = encode_labels(train_data)
    test_data['data'] = label_enc.transform(test_data['label'])

    x_train, x_test, extractor = extract_features(train_data['processed_text'], test_data['processed_text'])

    model = train_model(x_train, train_data['label'])

    # Evaluate model on training data
    print("EVALUATING ON TRAINING DATA")
    accuracy, report = evaluate_model(model, x_train, train_data['label'])
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

    # Evaluate model on test data
    print("EVALUATING ON TEST DATA")

if __name__ == "__main__":
    main()
