from gensim.parsing.preprocessing import STOPWORDS
import os
import re
import chardet
import email 
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from wordcloud import WordCloud
from bs4 import BeautifulSoup
from string import punctuation



from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report

from imblearn.over_sampling import RandomOverSampler
import nltk; nltk.download('popular')
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize


# Download necessary NLTK resources
nltk.download('popular')

# Define root directory for email dataset
ROOT_DIR = "/content/drive/MyDrive/SpamAssassinMessages/SpamAssassinMessages"

# Function to flag emails as spam (1) or ham (0)
def flag_emails(filepath, positive_indicator="spam"):
    return 1 if positive_indicator in filepath else 0

# Function to extract text from an email message
def extract_email_text(message):
    """Extracts text from email body, handling both single-part and multi-part emails."""
    email_text = ""
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_maintype() == "text":
                try:
                    email_text += part.get_payload(decode=True).decode(errors="ignore") + " "
                except Exception as e:
                    print(f"Error decoding email part: {e}")
    else:
        try:
            email_text = message.get_payload(decode=True).decode(errors="ignore")
        except Exception as e:
            print(f"Error decoding single-part email: {e}")
    return email_text.strip()

# Function to load emails from a directory
def load_emails(root_dir):
    """Loads all emails from the specified directory into a Pandas DataFrame."""
    email_data = {"text": [], "label": []}
    
    for dirpath, _, filenames in os.walk(root_dir, topdown=False):
        for name in filenames:
            filepath = os.path.join(dirpath, name)
            label = flag_emails(filepath)
            
            try:
                with open(filepath, "rb") as f:  # Reading in binary mode for better decoding
                    msg = email.message_from_bytes(f.read(), policy=email.policy.default)
                    email_text = extract_email_text(msg)
                    if email_text:  # Only add non-empty messages
                        email_data["text"].append(email_text)
                        email_data["label"].append(label)
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")

    return pd.DataFrame(email_data)

# Load email dataset
email_df = load_emails(ROOT_DIR)

# Display basic statistics
print(f"Dataset contains {email_df.shape[0]} emails")
print(email_df.head())

# Save to CSV (optional)
email_df.to_csv("processed_emails.csv", index=False) file_list = []
for root, dirs, files in os.walk("/content/drive/MyDrive/SpamAssassinMessages/SpamAssassinMessages/", topdown=False):
    for name in files:
        tmp = os.path.join(root,name)
        file_list.append(tmp)
    for item in dirs:
        print(item)
i=0

file_list_full = []
while i < len(file_list):
    file_list_full.append(file_list[i])
    i+=1



len(file_list_full) # # Download NLTK resources if not already downloaded
# nltk.download("stopwords")
# nltk.download("punkt")

import warnings
from bs4 import MarkupResemblesLocatorWarning

# Suppress the BeautifulSoup warning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def preprocess_text(text):
    """Cleans and tokenizes text, removing stopwords and applying stemming."""
    # Check if text contains HTML tags before parsing
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "html.parser").get_text()

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)

    # Tokenize words
    words = word_tokenize(text)

    # Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word not in STOPWORDS]

    return " ".join(words)

# Apply preprocessing to all emails
email_df["clean_text"] = email_df["text"].apply(preprocess_text)

# Display cleaned dataset
print(email_df.head())

# Save preprocessed data to CSV (optional)
email_df.to_csv("preprocessed_emails.csv", index=False)

 import string
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(email_df["clean_text"])  # Transform cleaned text
y = email_df["label"]

# Handle Class Imbalance with Oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Display processed dataset info
print(f"Original Dataset Shape: {email_df.shape}")
print(f"Resampled Dataset Shape: {X_resampled.shape}")

# Save processed data (optional)
df_vectorized = pd.DataFrame(X_resampled.toarray(), columns=vectorizer.get_feature_names_out())
df_vectorized["label"] = y_resampled
df_vectorized.to_csv("vectorized_emails.csv", index=False)


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize Naive Bayes classifier
nb_classifier = MultinomialNB()

# Perform Cross-Validation (10-Fold)
cv_scores = cross_val_score(nb_classifier, X_resampled, y_resampled, cv=10, scoring="accuracy")

# Train the model on the full dataset
nb_classifier.fit(X_resampled, y_resampled)

# Predict on the training set (since no separate test set yet)
y_pred = nb_classifier.predict(X_resampled)

# Print Cross-Validation Results
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Accuracy: {np.mean(cv_scores):.4f}")

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_resampled, y_pred))

# Print Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_resampled, y_pred))


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

# Initialize AdaBoost model
ada_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)

# Perform Grid Search for Hyperparameter Tuning
param_grid = {'n_estimators': [50, 100, 200]}
grid_search = GridSearchCV(ada_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model with Grid Search
grid_search.fit(X_resampled, y_resampled)

# Get the best parameters and best estimator
best_ada = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Predict on the training set
y_pred_ada = best_ada.predict(X_resampled)

# Print Classification Report
print("\nClassification Report (AdaBoost):")
print(classification_report(y_resampled, y_pred_ada))

# Print Confusion Matrix
print("\nConfusion Matrix (AdaBoost):")
print(confusion_matrix(y_resampled, y_pred_ada))


import re

# Common spam words
SPAM_TRIGGER_WORDS = {"free", "win", "winner", "money", "cash", "credit card", "viagra", "click", "subscribe"}

# Function to extract features
def extract_features(email_text):
    """Extracts numerical features from an email."""
    features = {}
    
    # Count uppercase words
    features["num_uppercase_words"] = sum(1 for word in email_text.split() if word.isupper())
    
    # Count exclamation marks
    features["num_exclamations"] = email_text.count("!")
    
    # Count links
    features["num_links"] = len(re.findall(r"https?://\S+", email_text))
    
    # Count spam trigger words
    features["num_spam_words"] = sum(1 for word in email_text.split() if word.lower() in SPAM_TRIGGER_WORDS)
    
    # Check if email is a reply or forward
    features["is_reply_forward"] = int("re:" in email_text.lower() or "fwd:" in email_text.lower())
    
    return features

# Apply feature extraction to all emails
feature_data = email_df["text"].apply(extract_features).apply(pd.Series)

# Merge extracted features into the main dataset
email_df = pd.concat([email_df, feature_data], axis=1)

# Display dataset with new features
print(email_df.head())

# Save feature-enhanced dataset
email_df.to_csv("feature_engineered_emails.csv", index=False)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Initialize models
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
xgboost = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)

# List of models to compare
models = {
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "XGBoost": xgboost
}

# Train & evaluate each model with cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring="accuracy")
    print(f"\n{name} - Accuracy: {np.mean(scores):.4f}")

    # Fit the model
    model.fit(X_resampled, y_resampled)

    # Predict on the training set
    y_pred = model.predict(X_resampled)

    # Print classification report
    print(f"\nClassification Report ({name}):")
    print(classification_report(y_resampled, y_pred))

    # Print confusion matrix
    print(f"\nConfusion Matrix ({name}):")
    print(confusion_matrix(y_resampled, y_pred))


from collections import defaultdict
import numpy as np

# Function to calculate log-likelihood ratios for words in spam vs ham
def calculate_log_likelihood(email_df):
    """Computes log-likelihood ratios for spam vs ham words."""
    word_counts = defaultdict(lambda: [0, 0])  # [ham_count, spam_count]

    # Iterate through emails and count word occurrences
    for text, label in zip(email_df["clean_text"], email_df["label"]):
        words = set(text.split())  # Unique words only
        for word in words:
            word_counts[word][label] += 1  # Increment count based on label

    # Compute log-likelihood ratios
    log_likelihoods = {}
    for word, (ham_count, spam_count) in word_counts.items():
        p_ham = (ham_count + 1) / (sum([c[0] for c in word_counts.values()]) + 1)  # Laplace smoothing
        p_spam = (spam_count + 1) / (sum([c[1] for c in word_counts.values()]) + 1)
        log_likelihoods[word] = np.log(p_spam / p_ham)

    return log_likelihoods

# Compute log-likelihoods
log_likelihoods = calculate_log_likelihood(email_df)

# Sort words by spam likelihood
sorted_spam_words = sorted(log_likelihoods.items(), key=lambda x: x[1], reverse=True)

# Display top spam-indicative words
print("\nTop Spam Words by Log-Likelihood:")
for word, score in sorted_spam_words[:20]:  # Top 20
    print(f"{word}: {score:.4f}")

# Save log-likelihood analysis
pd.DataFrame(sorted_spam_words, columns=["word", "log_likelihood"]).to_csv("spam_log_likelihoods.csv", index=False)


import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ----------------- 1. Sender Domain Probability -----------------
# Extract sender domains from email addresses
email_df["sender_domain"] = email_df["text"].str.extract(r'@([\w\.-]+)')

# Compute spam probability by domain
domain_counts = email_df.groupby("sender_domain")["label"].mean().sort_values(ascending=False)[:10]

# Plot bar chart
plt.figure(figsize=(14, 5))
sns.barplot(x=domain_counts.values, y=domain_counts.index, palette="muted")
plt.xlabel("Probability")
plt.ylabel("Feature")
plt.title("Top Sender Domains by Probability for Multinomial Naive Bayes")
plt.show()

# ----------------- 2. Spam vs. Ham Distribution -----------------
plt.figure(figsize=(5, 5))
email_df["label"].value_counts().plot.pie(autopct='%1.0f%%', labels=["Ham", "Spam"], colors=["#1f77b4", "#ff7f0e"])
plt.show()

# ----------------- 3. Word Cloud -----------------
text_data = " ".join(email_df["clean_text"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# ----------------- 4. Spam Keyword Probability -----------------
# Extract top spam-indicative words from log-likelihoods
top_spam_words = sorted_spam_words[:20]  # First 20 spam-heavy words
top_words, top_scores = zip(*top_spam_words)

plt.figure(figsize=(14, 5))
sns.barplot(x=top_scores, y=top_words, palette="muted")
plt.xlabel("Probability")
plt.ylabel("Feature")
plt.title("Top Email keywords by Probability for Multinomial Naive Bayes")
plt.show()


