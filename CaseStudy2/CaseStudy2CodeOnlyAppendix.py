# Code Block  1
import pandas as pd

# Define file paths
diabetic_data_path = r"D:\7333\Case Study 2\diabetic_data.csv"
ids_mapping_path = r"D:\7333\Case Study 2\IDs_mapping.csv"

# Load the datasets
diabetic_data = pd.read_csv(diabetic_data_path)
ids_mapping = pd.read_csv(ids_mapping_path)

# Display the first few rows
print("Diabetic Data Sample:")
print(diabetic_data.head())

print("\nIDs Mapping Data Sample:")
print(ids_mapping.head())

# Check dataset sizes
print(f"\nDiabetic Data Shape: {diabetic_data.shape}")
print(f"IDs Mapping Shape: {ids_mapping.shape}")



# Code Block  2
import pandas as pd

# Load datasets
diabetic_data_path = r"D:\7333\Case Study 2\diabetic_data.csv"
ids_mapping_path = r"D:\7333\Case Study 2\IDs_mapping.csv"

diabetic_data = pd.read_csv(diabetic_data_path)
ids_mapping = pd.read_csv(ids_mapping_path)

# Display basic info about the datasets
diabetic_data_info = diabetic_data.info()
ids_mapping_info = ids_mapping.info()

# Display first few rows of both datasets
diabetic_data_head = diabetic_data.head()
ids_mapping_head = ids_mapping.head()

diabetic_data_info, diabetic_data_head, ids_mapping_info, ids_mapping_head



# Code Block  3
# explore missing  data: 
import numpy as np

# Check for missing values
missing_values = diabetic_data.isin(['?', np.nan]).sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

# Display columns with missing values
print("Missing Values Summary:\n")
print(missing_values)

# Percentage of missing values
missing_percentage = (missing_values / diabetic_data.shape[0]) * 100
print("\nPercentage of Missing Values:\n")
print(missing_percentage)


# Code Block  4
# CHECK "READMITTED" CLASS DISTRIB. 

import seaborn as sns
import matplotlib.pyplot as plt

# Check target variable distribution
print("\nReadmitted Value Counts:\n")
print(diabetic_data['readmitted'].value_counts())

# Plot class distribution
sns.countplot(data=diabetic_data, x='readmitted', order=diabetic_data['readmitted'].value_counts().index)
plt.title("Readmission Distribution")
plt.show()


# Code Block  5
# CHECK FOR DUPLICATES

# Check for duplicate rows
duplicate_count = diabetic_data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")



# Code Block  6
# INSPECT FEATURE TYPES: 

# Display data types
print("\nData Types:\n")
print(diabetic_data.dtypes)




from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Drop high-missing-value columns
diabetic_data.drop(columns=['weight', 'max_glu_serum', 'A1Cresult', 'medical_specialty', 'payer_code'], inplace=True)

# Drop IDs (not useful for modeling)
diabetic_data.drop(columns=['encounter_id', 'patient_nbr'], inplace=True)

# Impute missing values for categorical features with the most frequent value
categorical_imputer = SimpleImputer(strategy="most_frequent")
diabetic_data[['race', 'diag_1', 'diag_2', 'diag_3']] = categorical_imputer.fit_transform(diabetic_data[['race', 'diag_1', 'diag_2', 'diag_3']])

# Convert 'readmitted' to numerical categories (0 = 'NO', 1 = '>30', 2 = '<30')
diabetic_data['readmitted'] = diabetic_data['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})

# Convert categorical features to numeric
label_encoders = {}
categorical_cols = ['race', 'gender', 'age', 'insulin', 'change', 'diabetesMed']

for col in categorical_cols:
    le = LabelEncoder()
    diabetic_data[col] = le.fit_transform(diabetic_data[col])
    label_encoders[col] = le

# Display cleaned dataset summary
print("\nCleaned Data Summary:\n")
print(diabetic_data.info())



# Code Block  7
import numpy as np

# Function to categorize ICD-9 codes
def categorize_diagnosis(code):
    try:
        code = float(code)
        if 390 <= code <= 459 or code == 785:
            return "Circulatory"
        elif 460 <= code <= 519 or code == 786:
            return "Respiratory"
        elif 520 <= code <= 579 or code == 787:
            return "Digestive"
        elif 580 <= code <= 629 or code == 788:
            return "Genitourinary"
        elif 250 <= code < 251:
            return "Diabetes"
        elif 800 <= code <= 999:
            return "Injury"
        else:
            return "Other"
    except:
        return "Other"

# Apply categorization to diagnosis columns
for col in ['diag_1', 'diag_2', 'diag_3']:
    diabetic_data[col] = diabetic_data[col].apply(categorize_diagnosis)

# One-hot encoding for diagnosis categories
diabetic_data = pd.get_dummies(diabetic_data, columns=['diag_1', 'diag_2', 'diag_3'], drop_first=True)

# Convert medication columns from categorical to numeric
medication_cols = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 
    'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
    'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 
    'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
    'metformin-rosiglitazone', 'metformin-pioglitazone'
]

for col in medication_cols:
    diabetic_data[col] = diabetic_data[col].map({'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3})

# Final check
print("\nFinal Processed Data Summary:\n")
print(diabetic_data.info())


# Re-encode 'insulin' column
diabetic_data['insulin'] = diabetic_data['insulin'].map({'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3})

# Convert Boolean Columns to Integer (0 or 1)
bool_cols = diabetic_data.select_dtypes(include=['bool']).columns
diabetic_data[bool_cols] = diabetic_data[bool_cols].astype(int)

# Verify changes
print(diabetic_data.info())



# Code Block  8
# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Ensure categorical columns are properly encoded
categorical_cols = diabetic_data.select_dtypes(include=['object']).columns.tolist()
numeric_cols = diabetic_data.select_dtypes(exclude=['object']).columns.tolist()

# One-Hot Encoding for categorical variables
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('scaler', StandardScaler(), numeric_cols)
])

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')

# Cross-Validation and Hyperparameter Tuning
param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],
    'logreg__solver': ['lbfgs', 'liblinear']
}

# Create a Pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('logreg', log_reg)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

# Fit Model
grid_search.fit(diabetic_data.drop(columns=['readmitted']), diabetic_data['readmitted'])

# Best Model Results
best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validated Accuracy:", grid_search.best_score_)

# Cross-validation predictions
y_pred = cross_val_predict(best_model, diabetic_data.drop(columns=['readmitted']), diabetic_data['readmitted'], cv=cv)
y_pred_prob = best_model.predict_proba(diabetic_data.drop(columns=['readmitted']))[:, 1]

# Confusion Matrix
conf_matrix = confusion_matrix(diabetic_data['readmitted'], y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(diabetic_data['readmitted'], y_pred)
print("Classification Report:\n", class_report)

# ROC-AUC Score
roc_auc = roc_auc_score(diabetic_data['readmitted'], y_pred_prob, multi_class='ovr')
print("ROC-AUC Score:", roc_auc)
- currently running 9:40pm 2-2-25   udsingh CPU  boo.  

######   Didn't get past above chunk.  I froze abd died of hypothermia here.  ran for 120 minutes then i stopped it and switched to gpu . 

###################################################################################################################################################################################################


1ST.  COLAB
2ND. LOCAL 

###################################################################################################################################################################################################
# GPU

# Ensure GPU usage 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load datasets from Google Drive
diabetic_data_path = '/content/drive/MyDrive/diabetic_data.csv'
ids_mapping_path = '/content/drive/MyDrive/IDs_mapping.csv'
diabetic_data = pd.read_csv(diabetic_data_path)
ids_mapping = pd.read_csv(ids_mapping_path)

# Display initial samples and shapes
print("Diabetic Data Sample:")
print(diabetic_data.head())
print("\nIDs Mapping Data Sample:")
print(ids_mapping.head())
print("\nDiabetic Data Shape:", diabetic_data.shape)
print("IDs Mapping Shape:", ids_mapping.shape)



# Display basic info about the datasets
print("\nDiabetic Data Info:")
print(diabetic_data.info())
print("\nIDs Mapping Info:")
print(ids_mapping.info())

# Explore missing data in diabetic_data
missing_values = diabetic_data.isin(['?', np.nan]).sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print("\nMissing Values Summary:")
print(missing_values)
missing_percentage = (missing_values / diabetic_data.shape[0]) * 100
print("\nPercentage of Missing Values:")
print(missing_percentage)

# Plot 'readmitted' class distribution
print("\nReadmitted Value Counts:")
print(diabetic_data['readmitted'].value_counts())
sns.countplot(data=diabetic_data, x='readmitted', order=diabetic_data['readmitted'].value_counts().index)
plt.title("Readmission Distribution")
plt.show()

# Drop unnecessary columns
diabetic_data.drop(columns=['weight', 'max_glu_serum', 'A1Cresult', 'medical_specialty', 'payer_code', 
                            'encounter_id', 'patient_nbr', 'description'], inplace=True)

# Fill missing values
for col in ['race', 'diag_1', 'diag_2', 'diag_3']:
    diabetic_data[col] = diabetic_data[col].fillna('Unknown')

# Convert 'readmitted' to numerical categories (Multi-Class Classification)
diabetic_data['readmitted'] = diabetic_data['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})

# Verify dataset after processing
diabetic_data.info(), diabetic_data.head()


# Check for duplicate rows
duplicate_count = diabetic_data.duplicated().sum()
print("\nNumber of duplicate rows:", duplicate_count)

# Inspect feature types
print("\nData Types:")
print(diabetic_data.dtypes)

# ----------------------------
# Data Cleaning and Preprocessing

# Drop columns with high missing values and ID columns
cols_to_drop = ['weight', 'max_glu_serum', 'A1Cresult', 'medical_specialty', 'payer_code', 'encounter_id', 'patient_nbr']
diabetic_data.drop(columns=cols_to_drop, inplace=True)



# Impute missing values for categorical features with the most frequent value.
# Replace '?' with np.nan then fill with mode.
categorical_imputer_cols = ['race', 'diag_1', 'diag_2', 'diag_3']
for col in categorical_imputer_cols:
    diabetic_data[col].replace('?', np.nan, inplace=True)
    diabetic_data[col].fillna(diabetic_data[col].mode()[0], inplace=True)

# Convert 'readmitted' to numerical categories (0 = 'NO', 1 = '>30', 2 = '<30')
diabetic_data['readmitted'] = diabetic_data['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})



# Convert selected categorical features to numeric using LabelEncoder
categorical_cols = ['race', 'gender', 'age', 'insulin', 'change', 'diabetesMed']
label_encoders = {}
for col in categorical_cols:


    le = LabelEncoder()
    diabetic_data[col] = le.fit_transform(diabetic_data[col])
    label_encoders[col] = le

print("\nCleaned Data Summary:")
print(diabetic_data.info())

# ----------------------------
# Further Processing: Categorize ICD-9 codes and encode medications



# Function to categorize ICD-9 diagnosis codes
def categorize_diagnosis(code):
    try:
        code = float(code)
        if 390 <= code <= 459 or code == 785:
            return "Circulatory"
        elif 460 <= code <= 519 or code == 786:
            return "Respiratory"
        elif 520 <= code <= 579 or code == 787:
            return "Digestive"
        elif 580 <= code <= 629 or code == 788:
            return "Genitourinary"
        elif 250 <= code < 251:
            return "Diabetes"
        elif 800 <= code <= 999:
            return "Injury"
        else:
            return "Other"
    except:
        return "Other"

# Apply categorization to diagnosis columns and then one-hot encode them
for col in ['diag_1', 'diag_2', 'diag_3']:
    diabetic_data[col] = diabetic_data[col].apply(categorize_diagnosis)
diabetic_data = pd.get_dummies(diabetic_data, columns=['diag_1', 'diag_2', 'diag_3'], drop_first=True)





# Convert medication columns from categorical to numeric values
medication_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
                   'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                   'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin',
                   'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
                   'metformin-rosiglitazone', 'metformin-pioglitazone']
for col in medication_cols:
    diabetic_data[col] = diabetic_data[col].map({'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3})

print("\nFinal Processed Data Summary:")
print(diabetic_data.info())

# Re-encode 'insulin' column if needed (ensuring it is numeric)
if diabetic_data['insulin'].dtype == object:
    diabetic_data['insulin'] = diabetic_data['insulin'].map({'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3})

    # Convert any Boolean columns to integers (0 or 1)
bool_cols = diabetic_data.select_dtypes(include=['bool']).columns
diabetic_data[bool_cols] = diabetic_data[bool_cols].astype(int)
print("\nVerified Data Types After Processing:")
print(diabetic_data.info())

# ----------------------------
# Prepare data for PyTorch model training

# Separate features and target variable
X = diabetic_data.drop(columns=['readmitted'])
y = diabetic_data['readmitted']

# Convert to numpy arrays (features as float32 and labels as int64)
X_np = X.values.astype(np.float32)
y_np = y.values.astype(np.int64)

# Split the data into training and testing sets (stratify by target)
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42, stratify=y_np)



# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the data to torch tensors and move them to the GPU
X_train_tensor = torch.tensor(X_train).to(device)
y_train_tensor = torch.tensor(y_train).to(device)
X_test_tensor = torch.tensor(X_test).to(device)
y_test_tensor = torch.tensor(y_test).to(device)

# ----------------------------

# Define a simple PyTorch model (a one-layer network for multi-class classification)
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear(x)

input_dim = X_train_tensor.shape[1]
output_dim = 3  # Three classes: 0, 1, 2
model = SimpleNN(input_dim, output_dim).to(device)
print("Model structure:", model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))

# ----------------------------


WAIT!!



--------

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

# Ensure GPU usage (this will use your A100 GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load datasets from Google Drive
diabetic_data_path = '/content/drive/MyDrive/diabetic_data.csv'
ids_mapping_path = '/content/drive/MyDrive/IDs_mapping.csv'
diabetic_data = pd.read_csv(diabetic_data_path)
ids_mapping = pd.read_csv(ids_mapping_path)

# Display initial samples and shapes
print("Diabetic Data Sample:")
print(diabetic_data.head())
print("\nIDs Mapping Data Sample:")
print(ids_mapping.head())
print("\nDiabetic Data Shape:", diabetic_data.shape)
print("IDs Mapping Shape:", ids_mapping.shape)


# Display basic info about the datasets
print("\nDiabetic Data Info:")
print(diabetic_data.info())
print("\nIDs Mapping Info:")
print(ids_mapping.info())

# Explore missing data in diabetic_data
missing_values = diabetic_data.isin(['?', np.nan]).sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print("\nMissing Values Summary:")
print(missing_values)
missing_percentage = (missing_values / diabetic_data.shape[0]) * 100
print("\nPercentage of Missing Values:")
print(missing_percentage)

# Plot 'readmitted' class distribution
print("\nReadmitted Value Counts:")
print(diabetic_data['readmitted'].value_counts())
sns.countplot(data=diabetic_data, x='readmitted', order=diabetic_data['readmitted'].value_counts().index)
plt.title("Readmission Distribution")
plt.show()


# Check for duplicate rows
duplicate_count = diabetic_data.duplicated().sum()
print("\nNumber of duplicate rows:", duplicate_count)

# Inspect feature types
print("\nData Types:")
print(diabetic_data.dtypes)

# ----------------------------
# Data Cleaning and Preprocessing

# Drop columns with high missing values and ID columns
cols_to_drop = ['weight', 'max_glu_serum', 'A1Cresult', 'medical_specialty', 'payer_code', 'encounter_id', 'patient_nbr']
diabetic_data.drop(columns=cols_to_drop, inplace=True)

# Impute missing values for categorical features with the most frequent value.
# Replace '?' with np.nan then fill with mode.
categorical_imputer_cols = ['race', 'diag_1', 'diag_2', 'diag_3']
for col in categorical_imputer_cols:
    diabetic_data[col].replace('?', np.nan, inplace=True)
    diabetic_data[col].fillna(diabetic_data[col].mode()[0], inplace=True)

# Convert 'readmitted' to numerical categories (0 = 'NO', 1 = '>30', 2 = '<30')
diabetic_data['readmitted'] = diabetic_data['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})

# Convert selected categorical features to numeric using LabelEncoder
categorical_cols = ['race', 'gender', 'age', 'insulin', 'change', 'diabetesMed']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    diabetic_data[col] = le.fit_transform(diabetic_data[col])
    label_encoders[col] = le

print("\nCleaned Data Summary:")
print(diabetic_data.info())

# Function to categorize ICD-9 diagnosis codes
def categorize_diagnosis(code):
    try:
        code = float(code)
        if 390 <= code <= 459 or code == 785:
            return "Circulatory"
        elif 460 <= code <= 519 or code == 786:
            return "Respiratory"
        elif 520 <= code <= 579 or code == 787:
            return "Digestive"
        elif 580 <= code <= 629 or code == 788:
            return "Genitourinary"
        elif 250 <= code < 251:
            return "Diabetes"
        elif 800 <= code <= 999:
            return "Injury"
        else:
            return "Other"
    except:
        return "Other"

# Apply categorization to diagnosis columns and then one-hot encode them
for col in ['diag_1', 'diag_2', 'diag_3']:
    diabetic_data[col] = diabetic_data[col].apply(categorize_diagnosis)
diabetic_data = pd.get_dummies(diabetic_data, columns=['diag_1', 'diag_2', 'diag_3'], drop_first=True)

# Convert medication columns from categorical to numeric values
medication_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
                   'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                   'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin',
                   'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
                   'metformin-rosiglitazone', 'metformin-pioglitazone']
for col in medication_cols:
    diabetic_data[col] = diabetic_data[col].map({'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3})

print("\nFinal Processed Data Summary:")
print(diabetic_data.info())

# Re-encode 'insulin' column if needed (ensuring it is numeric)
if diabetic_data['insulin'].dtype == object:
    diabetic_data['insulin'] = diabetic_data['insulin'].map({'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3})

# Convert any Boolean columns to integers (0 or 1)
bool_cols = diabetic_data.select_dtypes(include=['bool']).columns
diabetic_data[bool_cols] = diabetic_data[bool_cols].astype(int)
print("\nVerified Data Types After Processing:")
print(diabetic_data.info())



# Prepare data for PyTorch model training

# Separate features and target variable
X = diabetic_data.drop(columns=['readmitted'])
y = diabetic_data['readmitted']

# Convert to numpy arrays (features as float32 and labels as int64)
X_np = X.values.astype(np.float32)
y_np = y.values.astype(np.int64)

# Split the data into training and testing sets (stratify by target)
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42, stratify=y_np)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the data to torch tensors and move them to the GPU
X_train_tensor = torch.tensor(X_train).to(device)
y_train_tensor = torch.tensor(y_train).to(device)
X_test_tensor = torch.tensor(X_test).to(device)
y_test_tensor = torch.tensor(y_test).to(device)

# Calculate ROC-AUC Score for multi-class classification
# Compute softmax probabilities for the test outputs
probabilities = torch.softmax(test_outputs, dim=1).cpu().numpy()

# Replace NaN values with 0 in probabilities before calculating ROC-AUC
probabilities = np.nan_to_num(probabilities)

# Binarize the test labels for ROC-AUC calculation
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
roc_auc = roc_auc_score(y_test_binarized, probabilities, multi_class='ovr')
print("ROC-AUC Score:", roc_auc)





import cupy as cp
import cudf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


# diabetic_data = pd.read_csv("/content/drive/MyDrive/diabetic_data.csv")

# Convert to cuDF for GPU-based manipulations
diabetic_data_cudf = cudf.DataFrame(diabetic_data)

# List of categorical and numeric columns
categorical_cols = ['race', 'gender', 'age', 'change', 'diabetesMed', 'insulin']
numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

# OneHotEncoder without 'sparse=False' (for older scikit-learn)
ohe = OneHotEncoder(handle_unknown='ignore')
encoded_cats_sparse = ohe.fit_transform(diabetic_data_cudf[categorical_cols].to_pandas())

# Convert the sparse output to a dense NumPy array
encoded_cats_array = encoded_cats_sparse.toarray()

# Build a cuDF DataFrame of the encoded columns
encoded_cats_df_cudf = cudf.DataFrame(
    encoded_cats_array,
    columns=ohe.get_feature_names_out(categorical_cols)
)

# Drop original categorical columns, then concat the one-hot columns
diabetic_data_cudf.drop(columns=categorical_cols, inplace=True)
diabetic_data_cudf = cudf.concat([diabetic_data_cudf, encoded_cats_df_cudf], axis=1)

# Scale numeric columns
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(diabetic_data_cudf[numeric_cols].to_pandas())
scaled_numeric_cudf = cudf.DataFrame(scaled_numeric, columns=numeric_cols)

# Replace original numeric columns with scaled data
diabetic_data_cudf[numeric_cols] = scaled_numeric_cudf[numeric_cols]

# Separate features (X) and target (y)
X_cudf = diabetic_data_cudf.drop(columns=['readmitted'])
y_cudf = diabetic_data_cudf['readmitted']

# Convert cuDF to Pandas for scikit-learn
X = X_cudf.to_pandas()
y = y_cudf.to_pandas()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42)

# Logistic Regression in scikit-learn (CPU-based)
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)
y_pred_prob = log_reg.predict_proba(X_test)  # shape: (rows, #classes)

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# For multi-class, set multi_class='ovr' or 'ovo' as needed
roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
print("ROC-AUC Score:", roc_auc)






import shap
import numpy as np
import cudf
import cupy as cp
from cuml.ensemble import RandomForestClassifier

#  Convert cuDF to NumPy for SHAP analysis
X_train_np = X_train.to_pandas().to_numpy()
y_train_np = y_train.to_pandas().to_numpy()
X_test_np = X_test.to_pandas().to_numpy()

#  Train a GPU-based Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

print(" RandomForest Model Trained on GPU!")

#  Convert Model Predictions to Work with SHAP
def rf_predict(X):
    X_cudf = cudf.DataFrame(X)  # Convert NumPy to cuDF (GPU)
    return rf_model.predict(X_cudf).to_numpy()  # Convert back to NumPy

#  Initialize SHAP Explainer
explainer = shap.Explainer(rf_predict, X_train_np)

#  Compute SHAP Values
shap_values = explainer(X_test_np)

#  Aggregate Feature Importance
feature_importance = np.abs(shap_values.values).mean(axis=0)

#  Convert to cuDF DataFrame for Display
feature_importance_df = cudf.DataFrame({'Feature': X_train.columns.to_pandas(), 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

#  Display Feature Importance
import ace_tools as tools
tools.display_dataframe_to_user(name="Feature Importance Analysis", dataframe=feature_importance_df)
