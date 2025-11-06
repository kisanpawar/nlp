"""
Spam SMS Classification System
Single script: downloads dataset, preprocesses, trains models,
evaluates and saves the best model + vectorizer.
"""

import os
import re
import zipfile
import requests
import pandas as pd
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# === NLTK setup ===
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

# Make sure NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

STOPWORDS = set(stopwords.words('english'))
LEMMA = WordNetLemmatizer()

# === Utility: download & load dataset ===
def download_and_load_sms_spam():
    """
    Downloads the UCI SMS Spam Collection (smsspamcollection.zip)
    and returns a pandas DataFrame with columns: label, text
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    print("Downloading dataset...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(BytesIO(r.content))
    # The file inside is called "SMSSpamCollection"
    with z.open("SMSSpamCollection") as f:
        df = pd.read_csv(f, sep='\t', header=None, names=['label', 'text'])
    print("Loaded dataset: {} rows".format(len(df)))
    return df

# === Preprocessing ===
def preprocess_text(text):
    """
    Basic cleaning: lowercase, remove non-alphanum (except spaces),
    tokenize, remove stopwords, lemmatize, return cleaned string.
    """
    # lowercase
    text = text.lower()
    # replace URLs, phone numbers, emails (optional)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)  # remove numbers (optional)
    # remove punctuation except spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # tokenize
    tokens = word_tokenize(text)
    # remove stopwords and short tokens
    tokens = [t for t in tokens if (t not in STOPWORDS) and len(t) > 1]
    # lemmatize
    tokens = [LEMMA.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# === Main pipeline ===
def main():
    # 1) Download dataset
    df = download_and_load_sms_spam()

    # Quick inspection
    print(df.label.value_counts())

    # 2) Encode labels (ham=0, spam=1)
    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df['label'])  # ham->0, spam->1

    # 3) Preprocess texts (this may take a short while)
    print("Preprocessing texts...")
    df['clean_text'] = df['text'].apply(preprocess_text)

    # 4) Train/test split
    X = df['clean_text'].values
    y = df['label_enc'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)

    # 5) Feature extraction + models in pipelines
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df=2)

    # Model 1: Multinomial Naive Bayes
    nb_pipeline = make_pipeline(tfidf, MultinomialNB())

    # Model 2: Logistic Regression (stronger baseline)
    lr_pipeline = make_pipeline(tfidf, LogisticRegression(
        solver='liblinear', C=1.0, max_iter=1000))

    # 6) Train
    print("Training MultinomialNB...")
    nb_pipeline.fit(X_train, y_train)

    print("Training LogisticRegression...")
    lr_pipeline.fit(X_train, y_train)

    # 7) Evaluate
    for name, model in [("MultinomialNB", nb_pipeline), ("LogisticRegression", lr_pipeline)]:
        print("\n=== Evaluation:", name, "===")
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 8) Choose best model by accuracy on test set (simple)
    acc_nb = accuracy_score(y_test, nb_pipeline.predict(X_test))
    acc_lr = accuracy_score(y_test, lr_pipeline.predict(X_test))
    if acc_lr >= acc_nb:
        best_model = lr_pipeline
        best_name = "LogisticRegression"
        best_acc = acc_lr
    else:
        best_model = nb_pipeline
        best_name = "MultinomialNB"
        best_acc = acc_nb

    print(f"\nBest model: {best_name} with accuracy {best_acc:.4f}")

    # 9) Save model and label encoder
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(best_model, "saved_models/best_spam_model.joblib")
    joblib.dump(le, "saved_models/label_encoder.joblib")
    print("Saved model to saved_models/best_spam_model.joblib")

    # 10) Example predictions
    examples = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "Hey, are we still meeting for lunch today?",
        "CONGRATS! You won a free voucher. Call 09061790125 to claim ASAP."
    ]
    examples_clean = [preprocess_text(t) for t in examples]
    preds = best_model.predict(examples_clean)
    for txt, p in zip(examples, preds):
        label = le.inverse_transform([p])[0]
        print(f"Text: {txt}\n-> Predicted: {label}\n")

if __name__ == "__main__":
    main()
