from gensim.parsing.preprocessing import STOPWORDS
from nltk import word_tokenize
import os, chardet, email
import pandas as pd 
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import StratifiedKFold, GridSearchCV,  train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, completeness_score
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report
from nltk.corpus import stopwords
import nltk; nltk.download('popular')
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import re


file_list = []
for root, dirs, files in os.walk("/content/drive/MyDrive/SpamAssassinMessages/", topdown=False):
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



len(file_list_full)







%%capture
contents = []
types = []
labels = []
sender = []

i = 0

for root, dirs, files in os.walk("/content/drive/MyDrive/SpamAssassinMessages/"):
    for name in files:
        blob = open(os.path.join(root, name), 'rb').read()

        # Determining the encoding of the email
        result = chardet.detect(blob)
        encoding_dynamic = result['encoding']
        
        # If encoding detection fails, use a default encoding (e.g., utf-8)
        if encoding_dynamic is None:
            encoding_dynamic = 'utf-8'  # or 'latin-1', depending on your data

        with open(os.path.join(root, name), 'rb') as f:
            decoded_email = f.read().decode(encoding_dynamic, errors='replace')
            
            x = email.message_from_string(decoded_email)
            print('normal', os.path.join(root, name))
            i += 1

            # ... (rest of your code remains the same) ...

            #adding in the sender's domain as a new column
            sender_raw = x.get('From')
            if sender_raw:
                sender_domain = sender_raw.split('@')[-1]
                if sender_domain:
                    sender.append(sender_domain)
                else:
                    sender.append("missing")
            else:
                sender.append("missing")


            if "multipart" in x.get_content_type():
                content_parts = []
                for part in x.walk():
                    if part.get_content_type() == "text/plain":
                        decoded_part = part.get_payload(decode=True).decode(encoding_dynamic, errors='replace')
                        content_parts.append(decoded_part.replace("\n", " "))
                    elif part.get_content_type() == "text/html":
                        soup = BeautifulSoup(part.get_payload(decode=True).decode(encoding_dynamic, errors='replace'), 'html.parser')
                        decoded_part = soup.get_text().replace("\n", " ")
                        content_parts.append(decoded_part)
                content = "\n".join(content_parts)
                contents.append(content)
                types.append(x.get_content_type())

            elif "text/plain" in x.get_content_type():
                decoded_content = x.get_payload(decode=True).decode(encoding_dynamic, errors='replace')
                contents.append(decoded_content.replace("\n", " "))
                types.append(x.get_content_type())

            elif "text/html" in x.get_content_type():
                soup = BeautifulSoup(x.get_payload(decode=True).decode(encoding_dynamic, errors='replace'), 'html.parser')
                decoded_content = soup.get_text().replace("\n", " ")
                contents.append(decoded_content)
                types.append(x.get_content_type())

            if "ham" in root:
                labels.append(0)
            elif "spam" in root:
                labels.append(1)
            else:
                print("JESS STOP PROBLEM-- LABEL NOT FOUND")





print('contents length: ', len(contents))
print('types length: ', len(types))
print('labels length: ', len(labels))
print('file list length: ', len(file_list_full))
print('sender length: ', len(sender))


# Encode the contents array to utf-8, before it gets added to pandas dataframe
encoded_contents = [content.encode('utf-8', errors='replace') for content in contents]





data = pd.DataFrame({"Text":encoded_contents,"Label":labels, "Type":types, "Filename": file_list_full, "Sender": sender})
data


data.info()
data.head()
data['Text'] = data['Text'].astype('|S') # which will by default set the length to the max len it encounters
data.info()


counts = data['Label'].replace(to_replace = [0,1],value = ['Ham','Spam']).value_counts().plot(kind = 'bar',title = "Email Type Counts",ylabel = "Count")


special_chr = ['br','subject','spamassassin']
stop = list(stopwords.words('english')) + special_chr

# MultiNB
X = data['Text']
y = data['Label'].values

skf = StratifiedKFold(n_splits=10,random_state=610,shuffle=True)
skf.get_n_splits(X,y)

#need to vectorize within pipeline to prevent data leakage (otherwise tfidf would be fitted over all the data)
my_pipe = Pipeline([
    ('vectorizer',TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stop, ngram_range=(2,2), analyzer='word')),
    ('estimator',MultinomialNB())
])
pipe_params = {
    'estimator__alpha':[.01,.05,.1]
}

grid = GridSearchCV(my_pipe,pipe_params,cv=skf,scoring='accuracy',n_jobs=-1)
results = grid.fit(X,y)
print(results.best_estimator_)
print(results.best_score_)








results_df=pd.DataFrame(results.cv_results_)
results_df








skf = StratifiedKFold(n_splits=10,random_state=610,shuffle=True)
skf.get_n_splits(X,y)

#need to vectorize within pipeline to prevent data leakage (otherwise tfidf would be fitted over all the data)
my_pipe = Pipeline([
    ('vectorizer',TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stop, ngram_range=(2,2), analyzer='word')),
    ('estimator',MultinomialNB())
])
pipe_params = {
    'estimator__alpha':[.01,.05,.1]
}

grid = GridSearchCV(my_pipe,pipe_params,cv=skf,scoring='recall',n_jobs=-1)
results = grid.fit(X,y)
print(results.best_estimator_)
print(results.best_score_)
#running model on messages
X = data['Text']
y = data['Label'].values

# MultiNB with best params
my_pipe = Pipeline([
    ('vectorizer',TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stop, ngram_range=(2,2), analyzer='word')),
    ('estimator',MultinomialNB(alpha=.01))
])
acc_score = []
rec_score = []
feature_names = []

for i, (train,test) in enumerate(skf.split(X,y)):
    my_pipe.fit(X[train],y[train])
    p = my_pipe.predict(X[test])
    acc_score.append(accuracy_score(y[test],p))
    rec_score.append(recall_score(y[test],p))
    print(classification_report(y[test],p))

    #feature importance by word
    feature_log_probs = my_pipe.named_steps['estimator'].feature_log_prob_
    feature_names = my_pipe.named_steps['vectorizer'].get_feature_names_out()


# Convert feature log probabilities to probabilities
feature_probs = np.exp(feature_log_probs)

# Create a DataFrame to store feature importance
feature_probability = pd.DataFrame({'Feature': feature_names,
                                   'Probability': feature_probs[1]})

# Sort feature importance by importance score
feature_probability = feature_probability.sort_values(by='Probability', ascending=False)


# saving scores for comparison    
mNB_scores = pd.DataFrame({'accuracy':acc_score,
                           'recall':rec_score})

idx = feature_probability['Probability'].sort_values(ascending = False).head(20).index
feature_probability2 = feature_probability.loc[idx]

plt.style.use('ggplot')
plt.figure(figsize = (15,8))
sns.barplot(x='Probability',y='Feature',data=feature_probability2).set(title='Top Email keywords by Probability for Multinomial Naive Bayes')




plt.rcParams["figure.figsize"] = (7,7)
ConfusionMatrixDisplay.from_predictions(y[test],my_pipe.predict(X[test]),cmap='Blues')
plt.title("Confusion Matrix Multinomial NB for Email Text")
plt.grid(False)
plt.show()

#rerunning model on Sender Domain
X_s = data['Sender']
y = data['Label'].values

# MultiNB with best params
my_pipe = Pipeline([
    ('vectorizer',TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stop, ngram_range=(2,2), analyzer='word')),
    ('estimator',MultinomialNB(alpha=results.best_params_.get('estimator__alpha')))
])
acc_score = []
rec_score = []
feature_names = []

for i, (train,test) in enumerate(skf.split(X_s,y)):
    my_pipe.fit(X_s[train],y[train])
    p = my_pipe.predict(X_s[test])
    acc_score.append(accuracy_score(y[test],p))
    rec_score.append(recall_score(y[test],p))
    print(classification_report(y[test],p))

    #feature importance by word
    feature_log_probs = my_pipe.named_steps['estimator'].feature_log_prob_
    feature_names = my_pipe.named_steps['vectorizer'].get_feature_names_out()


# Convert feature log probabilities to probabilities
feature_probs = np.exp(feature_log_probs)

# Create a DataFrame to store feature importance
feature_probability = pd.DataFrame({'Feature': feature_names,
                                   'Probability': feature_probs[1]})

# Sort feature importance by importance score
feature_probability = feature_probability.sort_values(by='Probability', ascending=False)


# saving scores for comparison    
mNB_scores2 = pd.DataFrame({'accuracy':acc_score,
                           'recall':rec_score})




idx = feature_probability['Probability'].sort_values(ascending = False).head(10).index
feature_probability2 = feature_probability.loc[idx]

plt.style.use('ggplot')
plt.figure(figsize = (15,5))
sns.barplot(x='Probability',y='Feature',data=feature_probability2).set(title='Top Sender Domains by Probability for Multinomial Naive Bayes')


cm = confusion_matrix(y[test],p)
plt.rcParams["figure.figsize"] = (7,7)
dist =ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = ['Ham','Spam'])
dist.plot(cmap = "Blues")
plt.title("Confusion Matrix for Multinomial NB")
plt.grid(False)
plt.show()


# support vector machine - classifier
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.svm import SVC

my_pipe = Pipeline([
    ('vectorizer',TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, ngram_range=(2,2), analyzer='word')),
    ('estimator',SVC(random_state=610))
])
pipe_params = {
    'estimator__C':[.01,.1,.5,1],
    'estimator__gamma':['scale','auto']
}

grid = GridSearchCV(my_pipe,pipe_params,cv=skf,scoring='accuracy',n_jobs=-1)
results = grid.fit(X,y)
print(results.best_estimator_)
print(results.best_score_)


# svc with best params
my_pipe = Pipeline([
    ('vectorizer',TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, ngram_range=(2,2), analyzer='word')),
    ('estimator',SVC(random_state=610,C=1))#not returning best for gamma for some reason
])
acc_score = []
rec_score = []
for i, (train,test) in enumerate(skf.split(X,y)):
    my_pipe.fit(X[train],y[train])
    p = my_pipe.predict(X[test])
    acc_score.append(accuracy_score(y[test],p))
    rec_score.append(recall_score(y[test],p))
    print(classification_report(y[test],p))
    
svc_scores = pd.DataFrame({'accuracy':acc_score,
                           'recall':rec_score})

cm = confusion_matrix(y[test],p)
plt.rcParams["figure.figsize"] = (7,7)
dist =ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = ['Ham','Spam'])
dist.plot(cmap = "Blues")
plt.title("Confusion Matrix for SVC")
plt.grid(False)
plt.show()




# random forest classifier
from sklearn.ensemble import RandomForestClassifier


my_pipe = Pipeline([
    ('vectorizer',TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, ngram_range=(2,2), analyzer='word')),
    ('estimator',RandomForestClassifier(random_state=610,max_depth = 100))
])
pipe_params = {
    'estimator__n_estimators':[50,100,200],
    'estimator__class_weight':['balanced',None]
}

grid = GridSearchCV(my_pipe,pipe_params,cv=skf,scoring='accuracy',n_jobs=-1)
results = grid.fit(X,y)
print(results.best_estimator_)
print(results.best_score_)





# random forest classifier
from sklearn.ensemble import RandomForestClassifier
# rf best params
my_pipe = Pipeline([
    ('vectorizer',TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, ngram_range=(2,2), analyzer='word')),
    ('estimator',RandomForestClassifier(max_depth = 100,
                                        random_state=610,
                                        n_estimators = 200))
])
acc_score = []
rec_score = []
for i, (train,test) in enumerate(skf.split(X,y)):
    my_pipe.fit(X[train],y[train])
    p = my_pipe.predict(X[test])
    acc_score.append(accuracy_score(y[test],p))
    rec_score.append(recall_score(y[test],p))
    print(classification_report(y[test],p))
    
rf_scores = pd.DataFrame({'accuracy':acc_score,
                           'recall':rec_score})







scores_df = pd.concat([mNB_scores.assign(Model = "MultiNB",Fold = mNB_scores.index),
                                 svc_scores.assign(Model = "SVC", Fold = svc_scores.index),
                                 rf_scores.assign(Model = "RandomForest", Fold = rf_scores.index)
                                 ])
scores = scores_df.melt(id_vars=["Model"],value_vars=['accuracy','recall'],var_name='Metric')





import seaborn as sns
sns.boxplot(scores,x='Model',y='value',hue='Metric').set(title = 'Error Distribution by Model')


mNB_scores[['accuracy','recall']].describe().T




# clustering
## checking to see if KMeans can find two clusters that match fairly well with target

from sklearn.cluster import KMeans
clustering = KMeans(n_clusters=2,n_init='auto',random_state=610).fit(X)
km_labels = clustering.labels_







# clustering
from sklearn.cluster import DBSCAN #DBSCAN works with

clustering = DBSCAN(eps=1,n_jobs=-1).fit(X)
db_labels = clustering.labels_



# saved from previous run
mNB_scores = pd.read_csv('/content/mNB_scores.csv')
svc_scores = pd.read_csv('/content/svc_scores.csv')
rf_scores = pd.read_csv('/content/rf_scores.csv')


#######################################################################
#load data functions
# def flag_emails(dirpath, positive_indicator="spam"):
#     if positive_indicator in dirpath:
#         return 1
#     else:
#         return 0 
    
# def import_messages(root_dir="/content/drive/MyDrive/SpamAssassinMessages/SpamAssassinMessages", encoding="cp1256", positive_indicator="spam"):
    
#     messages = {"message":[], "label":[]}
    
#     for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
#         for name in filenames:
#             fullpath = os.path.join(dirpath, name)
#             messages['label'].append(flag_emails(dirpath=dirpath, positive_indicator=positive_indicator))
#             with open(fullpath,'r', encoding=encoding) as f:
#                 try:
#                     msg = email.message_from_file(f)
#                     messages['message'].append(msg)
#                 except UnicodeDecodeError as e:
#                     print(f"Error occured with encoding type: {encoding}\n{e}")
#                     return
                 
#     return messages

# def create_email_string(message):
#     msg_text = ""
#     for msg_part in message.walk():
#         if "text" in msg_part.get_content_type():
#             msg_text = msg_text + " " + msg_part.get_payload()
#     return msg_text

# def import_emails(root_dir="/content/drive/MyDrive/SpamAssassinMessages/SpamAssassinMessages", encoding="cp1256", positive_indicator="spam"):
    
#     messages = import_messages(root_dir=root_dir, 
#                                encoding=encoding, 
#                                positive_indicator=positive_indicator)
    
#     messages['text'] = [create_email_string(message=msg) for msg in messages['message']]

#     df = pd.DataFrame(messages)
#     first_cols = ['text', 'label']
#     df = df.loc[:, first_cols]

#     return df


# emails = import_emails(root_dir="content/drive/MyDrive/SpamAssassinMessages/SpamAssassinMessage", encoding="cp437", positive_indicator="spam")


# def remove_tags(html):
#     soup = BeautifulSoup(html, "html.parser")
#     for data in soup(['style', 'script']):
#         data.decompose()
#     return ' '.join(soup.stripped_strings)

# emails['text'] = emails['text'].apply(remove_tags)

# def clean_email(email):
#     email = re.sub("\d+", " ", email)
#     email = email.replace('\n', ' ')
#     email = email.translate(str.maketrans("", "", punctuation))
#     email = email.lower()
#     return email

# emails['text'] = emails['text'].apply(clean_email)

# def preproces_text(email):

#     words = ""
#     stemmer = SnowballStemmer("english")
#     email = email.split()
#     for word in email:
#         words = words + stemmer.stem(word) + " "
#     return words

# emails['text_stemmed'] = emails['text'].apply(preproces_text)




# emails['text_wo_sw'] = emails['text'].apply(remove_stop_words)



# df_spam = emails.loc[emails['label']==1]
# df_ham = emails.loc[emails['label']==0]


# spam_counter = Counter(" ".join(df_spam['text_wo_sw']).split()).most_common(100)
# spam_result = pd.DataFrame(spam_counter, columns=['Word', 'Frequency'])

# ham_counter = Counter(" ".join(df_ham['text_wo_sw']).split()).most_common(100)
# ham_result = pd.DataFrame(ham_counter, columns=['Word', 'Frequency'])
















##################################################

import os
import email
import re
import chardet
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score
from imblearn.over_sampling import RandomOverSampler
import nltk

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

# Define root directory for email dataset
ROOT_DIR = "/content/drive/MyDrive/SpamAssassinMessages/SpamAssassinMessages"

# Function to flag emails as spam (1) or ham (0)
def flag_emails(filepath, positive_indicator="spam"):
    return 1 if positive_indicator in filepath else 0

# Function to extract sender domain
def extract_sender_domain(msg):
    sender = msg.get('From', '')
    if sender and '@' in sender:
        return sender.split('@')[-1]
    return 'unknown'

# Function to extract text from an email message
def extract_email_text(msg):
    """Extracts text from email body, handling both single-part and multi-part emails."""
    email_text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "text":
                try:
                    email_text += part.get_payload(decode=True).decode(errors="ignore") + " "
                except:
                    continue
    else:
        try:
            email_text = msg.get_payload(decode=True).decode(errors="ignore")
        except:
            pass
    return email_text.strip()

# Function to preprocess text
def preprocess_text(text):
    """Cleans and tokenizes text, removing stopwords and applying stemming."""
    stemmer = SnowballStemmer("english")
    stopwords_set = set(stopwords.words("english"))
    
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^\w\s]", "", text.lower())
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stopwords_set]
    
    return " ".join(words)

# Load emails
def load_emails(root_dir):
    email_data = {"text": [], "label": [], "sender_domain": []}
    
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            filepath = os.path.join(dirpath, name)
            label = flag_emails(filepath)
            
            try:
                with open(filepath, "rb") as f:
                    msg = email.message_from_bytes(f.read())
                    email_text = extract_email_text(msg)
                    sender_domain = extract_sender_domain(msg)
                    
                    if email_text:
                        email_data["text"].append(email_text)
                        email_data["label"].append(label)
                        email_data["sender_domain"].append(sender_domain)
            except:
                continue

    return pd.DataFrame(email_data)

# Load and preprocess dataset
email_df = load_emails(ROOT_DIR)
email_df["clean_text"] = email_df["text"].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(email_df["clean_text"])
y = email_df["label"]

# Handle Class Imbalance
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Train Multinomial Naive Bayes Model
nb_classifier = MultinomialNB()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = GridSearchCV(nb_classifier, {"alpha": [0.01, 0.1, 1]}, cv=skf, scoring="accuracy", n_jobs=-1)
cv_scores.fit(X_resampled, y_resampled)

# Predictions
y_pred = cv_scores.best_estimator_.predict(X_resampled)

# Print Results
print("Best Naive Bayes Model:", cv_scores.best_estimator_)
print("Accuracy:", accuracy_score(y_resampled, y_pred))
print(classification_report(y_resampled, y_pred))
ConfusionMatrixDisplay.from_predictions(y_resampled, y_pred, cmap="Blues")
plt.title("Confusion Matrix - Naive Bayes")
plt.show()

# Log-Likelihood Ratio for Spam Words
def calculate_log_likelihood(email_df):
    """Computes log-likelihood ratios for spam vs ham words."""
    word_counts = defaultdict(lambda: [0, 0])
    
    for text, label in zip(email_df["clean_text"], email_df["label"]):
        words = set(text.split())
        for word in words:
            word_counts[word][label] += 1
    
    log_likelihoods = {}
    for word, (ham_count, spam_count) in word_counts.items():
        p_ham = (ham_count + 1) / (sum(c[0] for c in word_counts.values()) + 1)
        p_spam = (spam_count + 1) / (sum(c[1] for c in word_counts.values()) + 1)
        log_likelihoods[word] = np.log(p_spam / p_ham)
    
    return log_likelihoods

log_likelihoods = calculate_log_likelihood(email_df)
sorted_spam_words = sorted(log_likelihoods.items(), key=lambda x: x[1], reverse=True)

# Display top spam-indicative words
spam_words_df = pd.DataFrame(sorted_spam_words[:20], columns=["word", "log_likelihood"])
print(spam_words_df)

# Visualizations
plt.figure(figsize=(10, 5))
sns.barplot(x="log_likelihood", y="word", data=spam_words_df)
plt.title("Top Spam Words by Log-Likelihood")
plt.show()

# Word Cloud
spam_words = " ".join(email_df[email_df["label"] == 1]["clean_text"])
spam_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(spam_words)

plt.figure(figsize=(10, 5))
plt.imshow(spam_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Spam Emails")
plt.show()
