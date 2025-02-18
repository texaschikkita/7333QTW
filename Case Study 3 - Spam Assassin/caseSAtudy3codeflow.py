
import os
import zipfile
import random
import re
import quopri
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# I install NLTK and download needed resources
!pip install nltk
nltk.download('punkt')
nltk.download('stopwords')

# I mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 1. I set my zip file path
zip_file_path = "/content/drive/MyDrive/SpamAssassinMessages.zip"
extract_path = "/content/drive/MyDrive/extracted_files"

# 2. I extract the SpamAssassin ZIP
if not os.path.exists(extract_path):
    os.makedirs(extract_path)
try:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
except Exception as e:
    print("Error extracting:", e)

# 3. I examine the extracted folders
extracted_folders = os.listdir(extract_path)
print("Extracted top-level items:", extracted_folders)

# 4. I pick a sample file from each category to preview
sample_files = {}
for folder in extracted_folders:
    folder_path = os.path.join(extract_path, folder)
    if os.path.isdir(folder_path):
        files_in_this_folder = os.listdir(folder_path)
        if files_in_this_folder:
            sample_files[folder] = files_in_this_folder[0]

sample_contents = {}
for category, file_name in sample_files.items():
    fpath = os.path.join(extract_path, category, file_name)
    try:
        with open(fpath, "r", encoding="latin-1") as f:
            preview = f.readlines()[:15]
            sample_contents[category] = "".join(preview)
    except Exception as e:
        sample_contents[category] = str(e)
print("Sample file previews:\n", sample_contents)

# 5. I define a function to extract the body of each email
def extract_email_body(email_text):
    text_decoded = quopri.decodestring(email_text).decode(errors="ignore")
    lines = text_decoded.split("\n\n", 1)
    email_body = lines[1] if len(lines) > 1 else text_decoded
    soup = BeautifulSoup(email_body, "html.parser")
    email_body = soup.get_text(separator=" ", strip=True)
    email_body = re.sub(r"^(Message-Id:|X-Loop:|Sender:|Errors-To:).*", "", email_body, flags=re.MULTILINE)
    email_body = "\n".join(line.strip() for line in email_body.splitlines() if line.strip())
    return email_body

# 6. I loop over each folder/file to clean up the emails and store them
cleaned_emails = {}
for folder in extracted_folders:
    folder_path = os.path.join(extract_path, folder)
    if os.path.isdir(folder_path):
        files_in_this_folder = os.listdir(folder_path)
        # I may store all files from each folder or just one. Here I do them all.
        # For demonstration, I gather each folder's text into a list.
        folder_texts = []
        for f in files_in_this_folder:
            path_f = os.path.join(folder_path, f)
            try:
                with open(path_f, "r", encoding="latin-1") as emailfile:
                    raw_text = emailfile.read()
                    cleaned_body = extract_email_body(raw_text)
                    folder_texts.append(cleaned_body)
            except:
                pass
        cleaned_emails[folder] = folder_texts

# 7. I label the data. We have known categories from SpamAssassin
label_mapping = {
    "easy_ham": 0,
    "easy_ham_2": 0,
    "hard_ham": 0,
    "spam": 1,
    "spam_2": 1
}

# 8. I flatten everything into a dataframe with columns ["Category","Message","Label"]
all_rows = []
for category, list_of_messages in cleaned_emails.items():
    for msg in list_of_messages:
        if category in label_mapping:
            lab = label_mapping[category]
            all_rows.append((category, msg, lab))

df = pd.DataFrame(all_rows, columns=["Category","Message","Label"])
print("Data shape:", df.shape)
print(df["Category"].value_counts())

# 9. I define a text preprocessing function
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_email_text(txt):
    txt = re.sub(r"(unsubscribe|click here|buy now|special offer|limited time|free trial|guaranteed).*","", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s+"," ",txt).strip()
    txt = txt.lower()
    txt = re.sub(r"[^\w\s]","",txt)
    tokens = word_tokenize(txt)
    sw = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in sw]
    return " ".join(tokens)

df["CleanText"] = df["Message"].apply(preprocess_email_text)

# 10. I drop rows that are empty after cleaning
df = df[df["CleanText"].str.strip() != ""].reset_index(drop=True)
print("Data shape after dropping empty messages:", df.shape)

# 11. I vectorize with TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["CleanText"])
y = df["Label"].values

# 12. I check the class distribution
print("Class distribution:", pd.Series(y).value_counts())

# 13. If necessary, I do oversampling for small classes
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X, y)
print("New class distribution:", pd.Series(y_ros).value_counts())

# 14. I perform a wide alpha search with cross validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold

alpha_range = np.logspace(-6, 6, 20)
param_grid = {"alpha": alpha_range}
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(MultinomialNB(), param_grid=param_grid, scoring="accuracy", cv=cv_strategy)
grid_search.fit(X_ros, y_ros)

best_alpha = grid_search.best_params_["alpha"]
best_score = grid_search.best_score_
print("Best alpha found:", best_alpha)
print("Cross-validation accuracy with best alpha:", best_score)

# 15. I train a final model and evaluate
final_nb = MultinomialNB(alpha=best_alpha)
final_nb.fit(X_ros, y_ros)

# 16. I do cross_val_score to measure performance again
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(final_nb, X_ros, y_ros, cv=cv_strategy, scoring="accuracy")
print("Cross-validated accuracy scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

# 17. I produce a confusion matrix on the entire oversampled set
y_pred_full = final_nb.predict(X_ros)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print("Classification Report:\n", classification_report(y_ros, y_pred_full, target_names=["Ham","Spam"]))

cm = confusion_matrix(y_ros, y_pred_full)
print("Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(cm, display_labels=["Ham","Spam"])
disp.plot()

# 18. Optional: I can generate synthetic spam to further boost spam examples
def generate_synthetic_spam(num_samples=5):
    spam_templates = [
        "Congratulations! You've won a {prize}. Click {link}.",
        "Urgent: {account} requires your attention. Login at {link} to fix issues.",
        "Limited Offer: Grab {discount}% off on {product} at {link}.",
        "Dear {name}, get your free gift at {link} now!",
        "Exclusive deal: {cash} reward if you sign up at {link}.",
    ]
    synthetic_spam = []
    for _ in range(num_samples):
        temp = random.choice(spam_templates)
        spam_msg = temp.format(
            prize = random.choice(["$1,000","a free phone"]),
            link = random.choice(["http://fakepromo.com","http://scamlink.net"]),
            account = random.choice(["bank account","email account"]),
            discount = random.randint(20,90),
            product = random.choice(["a smartwatch","membership","software"]),
            name = random.choice(["Alex","Taylor","Chris"]),
            cash = random.choice(["$50","$100","$500"])
        )
        synthetic_spam.append(spam_msg)
    return synthetic_spam

# 19. This is the full code flow from top to bottom. If I want, I can run everything in order.