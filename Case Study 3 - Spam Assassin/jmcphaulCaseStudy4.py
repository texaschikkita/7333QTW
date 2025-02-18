# Ensure NLTK resources are available
!pip install nltk # Install NLTK if not already installed
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # Download 'punkt_tab' data

#  mount google drive

from google.colab import drive
drive.mount('/content/drive')




import os
import zipfile

# Path to the uploaded zip file
zip_file_path = "/content/drive/MyDrive/SpamAssassinMessages.zip"
extract_path = "content/drive/MyDrive/extracted_files"

# Extract the ZIP file
if not os.path.exists(extract_path):
    os.makedirs(extract_path)

try:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        extracted_files = os.listdir(extract_path)
except Exception as e:
    extracted_files = []
    print(f"Error extracting files: {e}")

# Display the extracted folder structure
extracted_files

# List extracted files to examine the dataset structure
extracted_files = os.listdir("content/drive/MyDrive/extracted_files")
print(extracted_files[:10])  # Display first 10 files for inspection


# Select a sample file from each category to inspect
sample_files = {}
for category in extracted_files:
    category_path = os.path.join(extract_path, category)
    if os.path.isdir(category_path):
        files = os.listdir(category_path)
        if files:
            sample_files[category] = files[0]  # Select the first file in the category

# Read the first few lines of each sample file
sample_contents = {}
for category, file in sample_files.items():
    file_path = os.path.join(extract_path, category, file)
    try:
        with open(file_path, "r", encoding="latin-1") as f:
            sample_contents[category] = "".join(f.readlines()[:15])  # Read first 15 lines
    except Exception as e:
        sample_contents[category] = f"Error reading file: {e}"

# Display sample contents
sample_contents


import re
import quopri
from bs4 import BeautifulSoup

def extract_email_body(email_text):
    """
    Extracts the body of an email by removing headers and decoding any encoded text.
    Handles quoted-printable encoding and removes HTML content.
    """
    # Decode quoted-printable encoding
    email_text = quopri.decodestring(email_text).decode(errors="ignore")

    # Remove headers (everything before the first blank line)
    lines = email_text.split("\n\n", 1)  # Split at the first blank line
    email_body = lines[1] if len(lines) > 1 else email_text

    # Remove HTML tags if the email contains HTML
    soup = BeautifulSoup(email_body, "html.parser")
    email_body = soup.get_text(separator=" ", strip=True)

    # Remove metadata fields that may persist
    email_body = re.sub(r"^(Message-Id:|X-Loop:|Sender:|Errors-To:).*", "", email_body, flags=re.MULTILINE)

    # Remove excessive whitespace and blank lines
    email_body = "\n".join([line.strip() for line in email_body.splitlines() if line.strip()])

    return email_body

# Apply extraction to each email category
cleaned_emails = {}
for category, file in sample_files.items():
    file_path = os.path.join(extract_path, category, file)
    try:
        with open(file_path, "r", encoding="latin-1") as f:
            email_text = f.read()
            cleaned_emails[category] = extract_email_body(email_text)
    except Exception as e:
        cleaned_emails[category] = f"Error reading file: {e}"

# Display extracted email bodies
cleaned_emails

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess the email text
def preprocess_email_text(text):
    """
    Preprocesses the email text by:
    - Removing unnecessary metadata and spam-specific artifacts.
    - Standardizing text casing and punctuation.
    - Removing stopwords.
    - Tokenizing text.
    """

    # Remove common promotional phrases in spam
    text = re.sub(r"(unsubscribe|click here|buy now|special offer|limited time|free trial|guaranteed).*", "", text, flags=re.IGNORECASE)

    # Remove excessive spaces and newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^\w\s]", "", text)

    # Tokenize and remove stop words
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    return " ".join(filtered_tokens)

# Apply preprocessing to all extracted emails
preprocessed_emails = {category: preprocess_email_text(content) for category, content in cleaned_emails.items()}

# Display a sample of the cleaned emails
preprocessed_emails

import pandas as pd # Import the pandas library and assign it to the name 'pd'

from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB

# Assign labels: 0 for Ham (Non-Spam), 1 for Spam
label_mapping = {
    "easy_ham": 0,
    "easy_ham_2": 0,
    "hard_ham": 0,
    "spam": 1,
    "spam_2": 1
}

# Convert dataset into a structured format
email_df = pd.DataFrame({ # Now 'pd' is recognized as pandas
    "Category": list(preprocessed_emails.keys()),
    "Message": list(preprocessed_emails.values()),
    "Label": [label_mapping[category] for category in preprocessed_emails.keys()]
})


from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB

# Assign labels: 0 for Ham (Non-Spam), 1 for Spam
label_mapping = {
    "easy_ham": 0,
    "easy_ham_2": 0,
    "hard_ham": 0,
    "spam": 1,
    "spam_2": 1
}

# Convert dataset into a structured format
email_df = pd.DataFrame({
    "Category": list(preprocessed_emails.keys()),
    "Message": list(preprocessed_emails.values()),
    "Label": [label_mapping[category] for category in preprocessed_emails.keys()]
})

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(email_df["Message"])
y = email_df["Label"]

# Initialize Naïve Bayes classifier
nb_classifier = MultinomialNB()

# Perform **cross-validation** using KFold (3 splits, shuffled data)
cv = KFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = cross_val_score(nb_classifier, X, y, cv=cv, scoring="accuracy")

# Display cross-validation accuracy results
cv_scores.mean(), cv_scores

# Check class distribution
email_df["Label"].value_counts()


from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid for alpha (Laplace smoothing)
param_grid = {"alpha": [0.01, 0.1, 0.5, 1, 5, 10]}

# Initialize Naïve Bayes classifier
nb_classifier = MultinomialNB()

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(nb_classifier, param_grid, cv=3, scoring="accuracy")
grid_search.fit(X, y)

# Get best parameters and best score
best_alpha = grid_search.best_params_["alpha"]
best_score = grid_search.best_score_

best_alpha, best_score


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Train final model using best alpha
final_nb = MultinomialNB(alpha=best_alpha)
final_nb.fit(X, y)

# Predict using cross-validation
y_pred = cross_val_score(final_nb, X, y, cv=3)

# Generate classification report
class_report = classification_report(y, final_nb.predict(X), target_names=["Ham", "Spam"])

# Generate confusion matrix
conf_matrix = confusion_matrix(y, final_nb.predict(X))

# Display confusion matrix
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["Ham", "Spam"])
disp.plot()

# Output classification report and confusion matrix
class_report, conf_matrix







import numpy as np # Import the numpy library and assign it to the alias 'np'
np.logspace(-6, 6, 20)  # 20 values between 10^-6 and 10^6







import numpy as np

# Count the number of samples per class
class_counts = np.bincount(y)

# Check if we can perform stratified k-fold with at least 3 splits
print("Class Counts:", class_counts)
print("Minimum Class Size:", class_counts.min())

# If min class size is < 3, we may need data augmentation
if class_counts.min() < 3:
    print("\n⚠ Warning: Some classes have fewer than 3 samples. Consider synthetic data augmentation.")

from imblearn.over_sampling import RandomOverSampler

# Apply Random Oversampling instead of SMOTE
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Check new class counts
new_class_counts = np.bincount(y_resampled)
print("New Class Counts After Random Oversampling:", new_class_counts)









from sklearn.model_selection import GridSearchCV
import numpy as np

# Define expanded hyperparameter grid for alpha
param_grid = {"alpha": np.logspace(-6, 6, 20)}  # Expanding search range

# Initialize Naïve Bayes classifier
nb_classifier = MultinomialNB()

# Perform Grid Search with Cross-Validation
# Change cv to a value less than or equal to the minimum number of samples in any class
grid_search = GridSearchCV(nb_classifier, param_grid, cv=2, scoring="accuracy")  # Changed cv to 2 
grid_search.fit(X, y)

# Get best parameters and best score
best_alpha = grid_search.best_params_["alpha"]
best_score = grid_search.best_score_

print(f"Best Alpha: {best_alpha}")
print(f"Best Accuracy: {best_score:.4f}")




from sklearn.model_selection import cross_val_score, StratifiedKFold

# Use Stratified K-Fold now that we have balanced classes
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Train and validate using cross-validation
nb_classifier = MultinomialNB(alpha=best_alpha)  # Use the best alpha found
cv_scores = cross_val_score(nb_classifier, X_resampled, y_resampled, cv=cv, scoring="accuracy")

# Print cross-validation results
print(f"\n Cross-Validation Accuracy Scores: {cv_scores}")
print(f" Mean Accuracy: {cv_scores.mean():.4f}")



from sklearn.model_selection import GridSearchCV

# Expanded hyperparameter grid for alpha
param_grid = {"alpha": np.logspace(-6, 6, 20)}  # Keep wide search range

# Initialize Naïve Bayes classifier
nb_classifier = MultinomialNB()

# Use StratifiedKFold with at least 3 splits
grid_search = GridSearchCV(nb_classifier, param_grid, cv=3, scoring="accuracy")
grid_search.fit(X_resampled, y_resampled)

# Get best parameters and best score
best_alpha = grid_search.best_params_["alpha"]
best_score = grid_search.best_score_

print(f" Best Alpha: {best_alpha}")
print(f" Best Accuracy After Cross-Validation: {best_score:.4f}")



from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Train final model using best alpha
final_nb = MultinomialNB(alpha=best_alpha)
final_nb.fit(X_resampled, y_resampled)

# Generate predictions
y_pred = final_nb.predict(X_resampled)

# Generate classification report
class_report = classification_report(y_resampled, y_pred, target_names=["Ham", "Spam"])

# Generate confusion matrix
conf_matrix = confusion_matrix(y_resampled, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["Ham", "Spam"])
disp.plot()

# Print classification report and confusion matrix
print("\n Classification Report:\n", class_report)
print("\n Confusion Matrix:\n", conf_matrix)


# generate synthetic data
import random

# List of common spam email topics
spam_templates = [
    "Congratulations! You've won a {prize}. Claim your reward now at {fake_link}.",
    "Dear {name}, your account has been compromised. Reset your password here: {fake_link}",
    "Limited time offer! Get a {discount}% discount on {product}. Buy now at {fake_link}",
    "Earn {money} per week working from home! Sign up at {fake_link}",
    "Urgent: Your bank account requires verification! Log in here: {fake_link}",
    "Exclusive deal just for you, {name}! Click {fake_link} to get {offer}",
    "Act now! Get {cashback}% cashback on all purchases. Offer expires soon!",
    "You have an unclaimed reward of {reward_amount}. Claim it before it expires!",
    "Get a FREE trial of {service}. No credit card required. Sign up today!",
    "Don't miss this opportunity! Invest in {crypto} and double your money instantly!"
]

# Function to generate synthetic spam emails
def generate_synthetic_spam(num_samples=10):
    synthetic_spam = []
    for _ in range(num_samples):
        template = random.choice(spam_templates)
        spam_email = template.format(
            name=random.choice(["John", "Jane", "Alex", "Chris"]),
            prize=random.choice(["$1,000", "a free iPhone", "a vacation to Hawaii"]),
            fake_link=random.choice(["http://scam.com", "http://fakepromo.com", "http://clickbait.com"]),
            discount=random.randint(10, 90),
            product=random.choice(["a new laptop", "a premium membership", "a smart watch"]),
            money=random.randint(500, 5000),
            offer=random.choice(["a VIP pass", "a free gift", "a limited-edition product"]),
            cashback=random.randint(5, 50),
            reward_amount=random.choice(["$50", "$100", "a mystery gift"]),
            service=random.choice(["VPN", "streaming service", "cloud storage"]),
            crypto=random.choice(["Bitcoin", "Ethereum", "Dogecoin"])
        )
        synthetic_spam.append(spam_email)
    
    return synthetic_spam

# Generate 10 synthetic spam emails
synthetic_spam_emails = generate_synthetic_spam(10)

# Print generated spam emails
for i, spam_email in enumerate(synthetic_spam_emails, 1):
    print(f" Synthetic Spam {i}: {spam_email}")


    # Append synthetic spam emails to the dataset
for spam_email in synthetic_spam_emails:
    # Use pd.concat instead of append
    email_df = pd.concat([email_df, pd.DataFrame([{"Category": "spam", "Message": spam_email, "Label": 1}])], ignore_index=True)

# Check class distribution again
print("\n Updated Class Distribution:\n", email_df["Label"].value_counts())


# Re-vectorize the text using TF-IDF
X_augmented = vectorizer.fit_transform(email_df["Message"])
y_augmented = email_df["Label"]

# Perform cross-validation
cv_scores = cross_val_score(nb_classifier, X_augmented, y_augmented, cv=3, scoring="accuracy")

# Display new cross-validation results
print("\n Cross-Validation Accuracy Scores:", cv_scores)
print(" Mean Accuracy After Augmentation:", cv_scores.mean())

# Retrain with best hyperparameter
final_nb = MultinomialNB(alpha=best_alpha)
final_nb.fit(X_augmented, y_augmented)

# Generate final classification report and confusion matrix
y_pred_augmented = final_nb.predict(X_augmented)
class_report_augmented = classification_report(y_augmented, y_pred_augmented, target_names=["Ham", "Spam"])
conf_matrix_augmented = confusion_matrix(y_augmented, y_pred_augmented)

# Display confusion matrix
disp = ConfusionMatrixDisplay(conf_matrix_augmented, display_labels=["Ham", "Spam"])
disp.plot()

# Output classification report
print("\n Final Classification Report After Augmentation:\n", class_report_augmented)


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Ensure dataset is prepared
X = X_augmented  # TF-IDF vectorized text data
y = y_augmented  # Labels

### **STEP 1: Stratified K-Fold Cross-Validation (10 folds)**
print("🔹 STEP 1: Stratified K-Fold Cross-Validation (10 folds)")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(MultinomialNB(alpha=best_alpha), X, y, cv=cv, scoring="accuracy")

mean_accuracy = np.mean(cv_scores)
std_dev = np.std(cv_scores)
confidence_interval = (mean_accuracy - 1.96 * (std_dev / np.sqrt(len(cv_scores))),
                       mean_accuracy + 1.96 * (std_dev / np.sqrt(len(cv_scores))))

print(f" Mean Accuracy: {mean_accuracy:.4f}")
print(f" Standard Deviation: {std_dev:.4f}")
print(f" 95% Confidence Interval: {confidence_interval}")

### **STEP 2: Split into Training & Test Sets, Evaluate on Unseen Data**
print("\n🔹 STEP 2: Train on 80% Data, Test on 20% Unseen Data")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model on 80% data
final_nb = MultinomialNB(alpha=best_alpha)
final_nb.fit(X_train, y_train)

# Predict on unseen test data
y_pred_test = final_nb.predict(X_test)

# Classification report on test data
class_report_test = classification_report(y_test, y_pred_test, target_names=["Ham", "Spam"])
print("\n Classification Report on Unseen Test Data:\n", class_report_test)

# Confusion Matrix on test data
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(conf_matrix_test, display_labels=["Ham", "Spam"])
disp.plot()
plt.title("Confusion Matrix on Test Data")
plt.show()

### **STEP 3: Apply K-Means Clustering for Pattern Analysis**
print("\n🔹 STEP 3: K-Means Clustering for Spam vs. Ham Separation")

# Apply K-Means clustering with 2 clusters (Spam vs. Ham)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X.toarray())  # Convert sparse matrix to dense for clustering

# Reduce dimensionality using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Reduce dimensionality using TSNE for better visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=5)  # Set perplexity to 5
X_tsne = tsne.fit_transform(X.toarray())

# Plot PCA results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="coolwarm", alpha=0.7)
plt.title("PCA: K-Means Clustering on Spam vs. Ham")

# Plot TSNE results
plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap="coolwarm", alpha=0.7)
plt.title("TSNE: K-Means Clustering on Spam vs. Ham")

plt.show()

print("\n All Steps Completed: Model Validated, Generalization Checked, Clusters Analyzed. Move on.")


 # 1. stratified k-fold CV
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

# Perform Stratified K-Fold Cross-Validation (10 folds)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_nb, X_augmented, y_augmented, cv=cv, scoring="accuracy")

# Compute Mean Accuracy, Standard Deviation, and 95% Confidence Interval
mean_accuracy = np.mean(cv_scores)
std_dev = np.std(cv_scores)
conf_interval = (mean_accuracy - 1.96 * std_dev / np.sqrt(len(cv_scores)),
                 mean_accuracy + 1.96 * std_dev / np.sqrt(len(cv_scores)))

print(f" Mean Accuracy: {mean_accuracy:.4f}")
print(f" Standard Deviation: {std_dev:.4f}")
print(f" 95% Confidence Interval: {conf_interval}")


# 2 train on 80% data test on 20% unseen data - confirms generalization by evaluating on separate test
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Split dataset into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42, stratify=y_augmented)

# Train the model on 80% data
final_nb.fit(X_train, y_train)

# Predict on unseen test data
y_pred_test = final_nb.predict(X_test)

# Generate classification report on test data
class_report_test = classification_report(y_test, y_pred_test, target_names=["Ham", "Spam"])
print("\n Classification Report on Unseen Test Data:\n", class_report_test)

# Generate and display confusion matrix
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(conf_matrix_test, display_labels=["Ham", "Spam"])
disp.plot()

# 3. 3: Apply K-Means Clustering for Pattern Analysis - uses unsupervbosed learning to analyze & verify spam characteristics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Apply K-Means Clustering (k=2 for spam vs. ham)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_augmented)

# Visualizing clusters using PCA (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_augmented.toarray())

# Scatter plot of PCA-reduced data
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="coolwarm")
plt.title("PCA: K-Means Clustering on Spam vs. Ham")

# Visualizing clusters using TSNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_augmented.toarray())

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=clusters, palette="coolwarm")
plt.title("TSNE: K-Means Clustering on Spam vs. Ham")
plt.show()





import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Define range of k values to test
k_values = range(2, 10)

# Store silhouette scores and inertia values
silhouette_scores = []
inertia_values = []

# Loop through different k values
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_augmented)
    
    silhouette_avg = silhouette_score(X_augmented, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    
    inertia_values.append(kmeans.inertia_)

# Plot the results
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Silhouette Score", color="blue")
ax1.plot(k_values, silhouette_scores, marker="o", linestyle="-", color="blue", label="Silhouette Score")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Inertia", color="red")
ax2.plot(k_values, inertia_values, marker="s", linestyle="--", color="red", label="Inertia")
ax2.tick_params(axis="y", labelcolor="red")

fig.tight_layout()
plt.title("Silhouette Score & Inertia vs. Number of Clusters")
plt.show()

# Determine the best k based on the highest silhouette score
best_k = k_values[np.argmax(silhouette_scores)]
print(f"Best k based on silhouette score: {best_k}")

# Re-run K-Means with the best k
optimal_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels_optimal = optimal_kmeans.fit_predict(X_augmented)

# Display final clustering results
print(f"Final K-Means Clustering Labels (Best k={best_k}):", np.unique(cluster_labels_optimal))



from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Set the best k
best_k = 9  # Update based on your results

# Apply K-Means with optimal k
optimal_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels_optimal = optimal_kmeans.fit_predict(X_augmented)

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_augmented.toarray())

# Apply TSNE for better separation
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_augmented.toarray())

# Plot PCA and TSNE Clustering Results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels_optimal, cmap='coolwarm', alpha=0.7)
ax1.set_title("PCA: K-Means Clustering with k=9")

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels_optimal, cmap='coolwarm', alpha=0.7)
ax2.set_title("TSNE: K-Means Clustering with k=9")

plt.show()


import numpy as np
import pandas as pd

# Create a DataFrame with cluster assignments
email_df["Cluster"] = cluster_labels_optimal

# Check how many spam and ham emails are in each cluster
cluster_summary = email_df.groupby(["Cluster", "Label"]).size().unstack(fill_value=0)
print("\n Cluster Purity Analysis:\n", cluster_summary)


# Compute mean TF-IDF values per cluster
tfidf_means = np.mean(X_augmented.toarray(), axis=0)
cluster_tfidf = pd.DataFrame(tfidf_means, index=vectorizer.get_feature_names_out())
print("\n Mean TF-IDF values per cluster:\n", cluster_tfidf)


from sklearn.metrics import adjusted_rand_score, silhouette_score

# Compute Adjusted Rand Index (ARI) - Measures how well the clusters match true labels
ari_score = adjusted_rand_score(y_augmented, cluster_labels_optimal)
print(f"\n Adjusted Rand Index (ARI): {ari_score:.4f}")

# Compute Silhouette Score - Measures how well-separated clusters are
silhouette = silhouette_score(X_augmented, cluster_labels_optimal)
print(f" Silhouette Score: {silhouette:.4f}")


# DBSCAN Clustering Cide

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Standardize the TF-IDF vectors (important for DBSCAN)
X_scaled = StandardScaler().fit_transform(X_augmented.toarray())

# Function to tune DBSCAN parameters
def tune_dbscan(X):
    eps_values = np.linspace(0.1, 2.0, 10)  # Try eps values from 0.1 to 2.0
    min_samples_values = [2, 3, 5, 10]  # Test different min_samples

    best_silhouette = -1
    best_params = None
    best_ari = None

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            unique_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Ignore noise

            if unique_clusters > 1:  # Only consider cases where clusters form
                sil_score = silhouette_score(X, labels)
                ari_score = adjusted_rand_score(y_augmented, labels)
                if sil_score > best_silhouette:
                    best_silhouette = sil_score
                    best_params = (eps, min_samples)
                    best_ari = ari_score

    return best_params, best_silhouette, best_ari

# Find best DBSCAN parameters
best_eps, best_min_samples = tune_dbscan(X_scaled)[0]
print(f"Best DBSCAN Parameters → eps: {best_eps}, min_samples: {best_min_samples}")

# Train DBSCAN with best parameters
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Evaluate DBSCAN Clustering
sil_score = silhouette_score(X_scaled, dbscan_labels)
ari_score = adjusted_rand_score(y_augmented, dbscan_labels)

print(f"DBSCAN Silhouette Score: {sil_score:.4f}")
print(f"DBSCAN Adjusted Rand Index (ARI): {ari_score:.4f}")

# Count clusters (excluding noise points)
cluster_counts = Counter(dbscan_labels)
print("DBSCAN Cluster Distribution:", cluster_counts)

# Visualization of DBSCAN Clusters using PCA & TSNE
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PCA Projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='coolwarm', alpha=0.7)
axes[0].set_title(f"PCA: DBSCAN Clustering (eps={best_eps}, min_samples={best_min_samples})")

# TSNE Projection
tsne = TSNE(n_components=2, perplexity=15, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=dbscan_labels, cmap='coolwarm', alpha=0.7)
axes[1].set_title(f"TSNE: DBSCAN Clustering (eps={best_eps}, min_samples={best_min_samples})")

plt.show()





import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter

# Define a range of hyperparameters to test
eps_values = np.arange(0.1, 1.1, 0.1)  # Test eps from 0.1 to 1.0
min_samples_values = range(2, 10)  # Test min_samples from 2 to 9

best_silhouette = -1  # Start with the lowest possible score
best_params = {}
best_labels = None

# Grid search for the best eps and min_samples
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_augmented.toarray())

        # Ignore cases where all points are noise (-1)
        if len(set(labels)) < 2:
            continue  

        # Compute silhouette score
        sil_score = silhouette_score(X_augmented, labels)
        
        # Track best parameters
        if sil_score > best_silhouette:
            best_silhouette = sil_score
            best_params = {"eps": eps, "min_samples": min_samples}
            best_labels = labels

# Print the best hyperparameters
print(f"Best DBSCAN Parameters → eps: {best_params['eps']}, min_samples: {best_params['min_samples']}")
print(f"Best DBSCAN Silhouette Score: {best_silhouette:.4f}")

# Apply DBSCAN with best parameters
final_dbscan = DBSCAN(eps=best_params["eps"], min_samples=best_params["min_samples"])
final_labels = final_dbscan.fit_predict(X_augmented.toarray())

# Cluster distribution
cluster_counts = Counter(final_labels)
print(f"DBSCAN Final Cluster Distribution: {cluster_counts}")

# Reduce dimensions using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_augmented.toarray())

# Plot DBSCAN clusters using PCA
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, cmap='coolwarm', alpha=0.7)
plt.colorbar(scatter)
plt.title(f"DBSCAN Clustering (eps={best_params['eps']}, min_samples={best_params['min_samples']})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
  import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Compute the hierarchical clustering linkage matrix
linked = linkage(X_augmented.toarray(), method='ward')

# Plot dendrogram to determine the best number of clusters
plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Try different numbers of clusters (k=2 to k=10)
best_k = None
best_silhouette = -1
best_labels = None

for k in range(2, 11):
    hierarchical = AgglomerativeClustering(n_clusters=k)
    labels = hierarchical.fit_predict(X_augmented.toarray())
    
    # Compute silhouette score
    sil_score = silhouette_score(X_augmented, labels)
    
    if sil_score > best_silhouette:
        best_silhouette = sil_score
        best_k = k
        best_labels = labels

print(f"Best Number of Clusters (k): {best_k}")
print(f"Best Silhouette Score: {best_silhouette:.4f}")

# Final model using best_k
final_hierarchical = AgglomerativeClustering(n_clusters=best_k)
final_labels = final_hierarchical.fit_predict(X_augmented.toarray())

# Reduce dimensions for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_augmented.toarray())

# Plot PCA-based clustering results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=final_labels, palette='coolwarm')
plt.title(f"Agglomerative Clustering with k={best_k}")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()



##AND HERE THE TANGENT BEGINS
# Ensmeble for Classification / Naive Bayes (ada boost) mean accuracy across 10 folds, check variarance with std dev. 95% ciS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Split into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Initialize Naive Bayes with AdaBoost
adaboost_nb = AdaBoostClassifier(estimator=MultinomialNB(alpha=best_alpha), n_estimators=50, random_state=42)
# Train model on training set
adaboost_nb.fit(X_train, y_train)

# Predict on test set
y_pred = adaboost_nb.predict(X_test)

# Generate classification report
class_report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["Ham", "Spam"])
disp.plot()
plt.title("Confusion Matrix on Test Data")
plt.show()

# Print classification report
print("\n Final Model Evaluation on Test Data:")
print(class_report)
