# For google module mounting within local env (wsl) 
```(python)
# don't need right nopw: 
# from google.oauth2 import service_account  
# from googleapiclient.discovery import build  

# # Replace with the path to your credentials.json  
# creds = service_account.Credentials.from_service_account_file("path/to/credentials.json")  

# service = build('drive', 'v3', credentials=creds)  

# # List files in your Google Drive  
# results = service.files().list(pageSize=10, fields="files(id, name)").execute()  
# items = results.get('files', [])  

# if not items:  
#     print('No files found.')  
# else:  
#     print('Files:')  
#     for item in items:  

# from google.colab import drive

# drive.mount('/content/drive')
```



```(python)

print("\nDiabetic Data Info:")
print(diabetic_data.info())

print("\nFirst few rows of diabetic_data:")
print(diabetic_data.head())

print("\nMissing values in dataset:")
missing_counts = diabetic_data.isnull().sum()
print(missing_counts[missing_counts > 0])

```

# 1. test and imports
import torch
import cupy as cp

# Check PyTorch CUDA availability
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version (PyTorch): {torch.version.cuda}")

# Check CuPy CUDA availability
print(f"CuPy CUDA available: {cp.cuda.is_available()}")
if cp.cuda.is_available():
    print(f"CUDA Version (CuPy): {cp.cuda.runtime.runtimeGetVersion() / 1000}")
    
    import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is NOT available.")

import cudf

print("cuDF is successfully installed!")
df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df)

# 2. EDA
import cudf  


# Load the data into cuDF DataFrames  
diabetic_data = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/diabetic_data.csv")  
ids_mapping = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/IDs_mapping.csv")    

# Ensure all string columns are treated as string type
diabetic_data = diabetic_data.astype(str)

# Replace '?' with None before converting to cuDF's NA
diabetic_data = diabetic_data.replace({'?': None}).fillna(cudf.NA)


# or (if needed) fix = convert only object clumns
# for col in diabetic_data.select_dtypes(include=['object']):
    # diabetic_data[col] = diabetic_data[col].replace({'?': None}).fillna(cudf.NA)


# Display dataset info  
print("\n Diabetic Data Info:")  
print(diabetic_data.info())  

print("\n First few rows of diabetic_data:")  
print(diabetic_data.head())  

print("\n IDs Mapping Data Info:")  
print(ids_mapping.info())  

print("\n First few rows of IDs_mapping:")  
print(ids_mapping.head())  

# Check missing values  
print("\n Missing values in dataset:")  
missing_counts = diabetic_data.isnull().sum()  
print(missing_counts[missing_counts > 0])

#. 2.b  
# Convert columns back to proper types: 
for col in diabetic_data.columns:
    if diabetic_data[col].str.isnumeric().all():
        diabetic_data[col] = diabetic_data[col].astype("int64")
# Convert columns back to proper types: 
for col in diabetic_data.columns:
    if diabetic_data[col].str.isnumeric().all():
        diabetic_data[col] = diabetic_data[col].astype("int64")



#  Step 2: Fix Data Types and Handle Missing Values

import cudf

#  Convert Numeric Columns First
numeric_cols = [
    "encounter_id", "patient_nbr", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses"
]

for col in numeric_cols:
    diabetic_data[col] = diabetic_data[col].astype("int64")

#  Convert Categorical Columns to String and Replace Missing Values
categorical_cols = [
    "race", "gender", "age", "payer_code", "medical_specialty",
    "diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
    "change", "diabetesMed", "readmitted"
]

for col in categorical_cols:
    diabetic_data[col] = diabetic_data[col].astype("str").replace({'?': cudf.NA})

#  Verify Fix
print(" Data Types Fixed and Missing Values Handled!")
print(diabetic_data.dtypes)



# Merge ids
 
import cudf

#  Check for non-numeric values
invalid_values = ids_mapping[~ids_mapping["admission_type_id"].str.isnumeric()]
print(" Non-Numeric Values in `admission_type_id`:\n", invalid_values)

#  Convert numeric values to integers
ids_mapping = ids_mapping[ids_mapping["admission_type_id"].str.isnumeric()]
ids_mapping["admission_type_id"] = ids_mapping["admission_type_id"].astype("int64")

print("\n Cleaned `ids_mapping` Data:")
print(ids_mapping.head())



# 3.2

#  Merge `diabetic_data` with `ids_mapping` on 'admission_type_id'
diabetic_data = diabetic_data.merge(ids_mapping, how="left", on="admission_type_id")

#  Drop unnecessary columns
columns_to_drop = [
    "weight", "max_glu_serum", "A1Cresult", "medical_specialty", "payer_code",
    "encounter_id", "patient_nbr", "description"  # 'description' is from ids_mapping
]
diabetic_data = diabetic_data.drop(columns=columns_to_drop)

#  Fill Missing Values in Key Categorical Columns
for col in ["race", "diag_1", "diag_2", "diag_3"]:
    diabetic_data[col] = diabetic_data[col].fillna("Unknown")

#  Convert 'readmitted' to numerical categories
diabetic_data["readmitted"] = diabetic_data["readmitted"].map({"NO": 0, ">30": 1, "<30": 2})

#  Verify Merge & Cleaning
print(" Merge Completed and Data Cleaned!")
print(diabetic_data.dtypes)
print("\n First Few Rows of Cleaned Data:")
print(diabetic_data.head())


# 4 Feature Engineering 

from cuml.preprocessing import StandardScaler

#  Define Numeric Columns
numeric_cols = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]

#  Initialize Scaler
scaler = StandardScaler()

#  Scale numeric features
diabetic_data[numeric_cols] = scaler.fit_transform(diabetic_data[numeric_cols])

#  Verify Scaling
print(" Numeric Features Scaled Successfully!")
print("First Few Rows:\n", diabetic_data.head())


### NOTE ON COLAB: CODE VARIATOPN: 
# Install cuML  
!pip install cuml-cu11 --index-url https://pypi.nvidia.com  

import cudf  
import pandas as pd  
from cuml.preprocessing import StandardScaler  

# Sample data (replace with your actual data loading)  
data = {'time_in_hospital': [3, 2, 5, 1, 4],   
        'num_lab_procedures': [59, 44, 70, 11, 44],   
        'num_procedures': [0, 5, 6, 2, 1],   
        'num_medications': [18, 13, 16, 8, 17],   
        'number_outpatient': [0, 2, 1, 0, 0],   
        'number_emergency': [0, 0, 0, 0, 0],   
        'number_inpatient': [0, 1, 0, 0, 0],   
        'number_diagnoses': [9, 6, 7, 5, 9]}  
diabetic_data = cudf.DataFrame(data)  


# Define Numeric Columns  
numeric_cols = [  
    "time_in_hospital", "num_lab_procedures", "num_procedures",  
    "num_medications", "number_outpatient", "number_emergency",  
    "number_inpatient", "number_diagnoses"  
]  

# Initialize Scaler  
scaler = StandardScaler()  

# Scale numeric features  
diabetic_data[numeric_cols] = scaler.fit_transform(diabetic_data[numeric_cols])  

# Verify Scaling  
print(" Numeric Features Scaled Successfully!")  
print("First Few Rows:\n", diabetic_data.head())


# load from saved: 
import joblib
import cudf

# Load trained model
log_reg = joblib.load("/home/jmcphaul/WSL_Case Study 2/logistic_regression_model.pkl")

# Load processed data
X_train = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/X_train_final.csv")
y_train = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/y_train_final.csv")
X_test = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/X_test_final.csv")
y_test = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/y_test_final.csv")

print("Reloaded saved model and data. Ready to continue.")


from cuml.preprocessing import OneHotEncoder

# Define categorical columns
categorical_cols = ["race", "gender", "age", "change", "diabetesMed", "insulin"]

# Initialize OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Apply OHE
encoded_cats = ohe.fit_transform(diabetic_data[categorical_cols])

# Manually generate column names
ohe_columns = []
for col in categorical_cols:
    unique_vals = diabetic_data[col].unique().to_arrow().to_pylist()  #  FIXED HERE
    ohe_columns.extend([f"{col}_{val}" for val in unique_vals])

# Convert encoded data to cuDF DataFrame
encoded_cats_df = cudf.DataFrame(encoded_cats, columns=ohe_columns)

# Drop original categorical columns and merge encoded data
diabetic_data.drop(columns=categorical_cols, inplace=True)
diabetic_data = cudf.concat([diabetic_data, encoded_cats_df], axis=1)

print(" One-Hot Encoding Completed Successfully!")
print(diabetic_data.head())


print(diabetic_data['readmitted'].dtype)
print(diabetic_data['readmitted'].unique())
non_numeric_cols = diabetic_data.drop(columns=['readmitted']).select_dtypes(exclude=['number']).columns

print(" Non-Numeric Columns in X:", non_numeric_cols)



# #  Convert 'diag_1', 'diag_2', 'diag_3' to categorical codes
# for col in ['diag_1', 'diag_2', 'diag_3']:
# 	# Convert 'admission_type_id' to numeric (shouldn't be object)
# diabetic_data['admission_type_id'] = diabetic_data['admission_type_id'].astype(int)

# # Drop 'description' column (unuseful for modeling)
# diabetic_data.drop(columns=['description'], inplace=True)

# # Confirm all columns are now numeric
# print("\n Updated column data types (ALL should be numeric now):")
# print(diabetic_data.dtypes)

#     diabetic_data[col] = diabetic_data[col].astype('category').cat.codes

# #  Convert all medication columns to binary (0/1)
# medication_cols = [
#     'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
#     'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
#     'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
#     'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 
#     'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
# ]

# for col in medication_cols:
#     diabetic_data[col] = (diabetic_data[col] != "No").astype("int32")  # 1 if "Yes", else 0

# #  Drop the 'description' column (not useful for training)
# diabetic_data.drop(columns=['description'], inplace=True)

# #  Convert everything to float32 (for GPU compatibility)
# diabetic_data = diabetic_data.astype("float32")

# print(" All Features Converted to Numeric Format!")


#  Convert 'diag_1', 'diag_2', 'diag_3' to categorical codes
for col in ['diag_1', 'diag_2', 'diag_3']:
    diabetic_data[col] = diabetic_data[col].astype('category').cat.codes

#  Convert all medication columns to binary (0/1)
medication_cols = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
    'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
]

for col in medication_cols:
    # Convert only if the column is of string type
    if diabetic_data[col].dtype == 'object':
        diabetic_data[col] = (diabetic_data[col].astype(str) != "No").astype("int32")  

#  Drop the 'description' column **if it exists**
if 'description' in diabetic_data.columns:
    diabetic_data.drop(columns=['description'], inplace=True)

#  Convert everything to float32 (for GPU compatibility)
diabetic_data = diabetic_data.astype("float32")

print(" All Features Converted to Numeric Format!")



from cuml.model_selection import train_test_split

#  Define Features (X) and Target (y)
X = diabetic_data.drop(columns=['readmitted'])
y = diabetic_data['readmitted'].astype("int32")  # Ensure target is integer

#  Split Data (Still in GPU)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(" Train/Test Split Completed! Shapes:")
print(f"  - X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  - X_test: {X_test.shape}, y_test: {y_test.shape}")


```(md)
Train/Test Split Successful!

X_train Shape: (732716, 63)
X_test Shape: (183178, 63)
y_train Shape: (732716,)
y_test Shape: (183178,)
```

from cuml.linear_model import LogisticRegression

#  Initialize and Train Model on GPU
log_reg = LogisticRegression(max_iter=1000, tol=1e-4)
log_reg.fit(X_train, y_train)

print(" Logistic Regression Model Trained Successfully!")


from cuml.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# Predict on Test Data
y_pred = log_reg.predict(X_test)

# Convert Predictions to CPU for Evaluation
y_test_cpu = y_test.to_pandas()
y_pred_cpu = y_pred.to_pandas()

# Compute Evaluation Metrics
accuracy = accuracy_score(y_test_cpu, y_pred_cpu)
conf_matrix = confusion_matrix(y_test_cpu, y_pred_cpu)
class_report = classification_report(y_test_cpu, y_pred_cpu)

# Display Results
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)


# Check Clas Imbalance: 

print("Class Distribution in Training Data:")
print(y_train.value_counts())

print("Class Distribution in Testing Data:")
print(y_test.value_counts())



from cuml.linear_model import LogisticRegression

class_weights = {0: 1.0, 1: 1.5, 2: 3.0}  # Increase weight for class 2
log_reg = LogisticRegression(penalty='l2', C=1.0, class_weight=class_weights)
log_reg.fit(X_train, y_train)


log_reg = LogisticRegression(penalty='l2', C=0.1, class_weight={0: 1.0, 1: 1.5, 2: 3.0})
log_reg.fit(X_train, y_train)

# Last attempt with qn before switching to sckkit: 
solver='qn'
log_reg = LogisticRegression(penalty='l2', C=1.0, class_weight={0: 1.0, 1: 1.5, 2: 3.0}, solver='qn')
log_reg.fit(X_train, y_train)

# F it
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    penalty='l2',
    C=1.0,
    class_weight={0: 1.0, 1: 1.5, 2: 3.0},
    solver='saga',
    max_iter=200,  # Reduce iterations
    warm_start=True  # Continue from the last iteration
)

for i in range(5):  # Train in smaller steps
    log_reg.fit(X_train.to_pandas(), y_train.to_pandas())
    print(f"Iteration {i+1} complete")


# Get metrics and analyze model: 


from sklearn.metrics import accuracy_score

y_pred = log_reg.predict(X_test.to_pandas())  # Predict on test data
accuracy = accuracy_score(y_test.to_pandas(), y_pred)

print(f"Accuracy: {accuracy:.4f}")


# Metrics and Analysis 

# Accuracy Score 
from sklearn.metrics import accuracy_score

y_pred = log_reg.predict(X_test.to_pandas())  # Predict on test data
accuracy = accuracy_score(y_test.to_pandas(), y_pred)

print(f"Accuracy: {accuracy:.4f}")



# Confusion Matrix
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test.to_pandas(), y_pred)
print("Confusion Matrix:\n", conf_matrix)


# Classification Report

from sklearn.metrics import classification_report

report = classification_report(y_test.to_pandas(), y_pred)
print("Classification Report:\n", report)


# Feature Importance: 
import pandas as pd
import numpy as np

feature_importance = np.abs(log_reg.coef_).flatten()
features = X_train.columns.to_pandas()

importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("Top Features Contributing to Predictions:")
print(importance_df.head(10))

 

 import pandas as pd
import numpy as np

# Get absolute importance values by taking mean across all classes
feature_importance = np.mean(np.abs(log_reg.coef_), axis=0)

# Ensure the number of features matches the dataset
if len(feature_importance) != len(X_train.columns):
    print("Warning: Feature importance length mismatch! Adjusting.")
    features = X_train.columns[:len(feature_importance)]  # Adjust length
else:
    features = X_train.columns

# Create DataFrame
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("Top Features Contributing to Predictions:")
print(importance_df.head(10))
# Get metrics and analyze model: 


from sklearn.metrics import accuracy_score

y_pred = log_reg.predict(X_test.to_pandas())  # Predict on test data
accuracy = accuracy_score(y_test.to_pandas(), y_pred)

print(f"Accuracy: {accuracy:.4f}")


# Metrics and Analysis 

# Accuracy Score 
from sklearn.metrics import accuracy_score

y_pred = log_reg.predict(X_test.to_pandas())  # Predict on test data
accuracy = accuracy_score(y_test.to_pandas(), y_pred)

print(f"Accuracy: {accuracy:.4f}")



# Confusion Matrix
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test.to_pandas(), y_pred)
print("Confusion Matrix:\n", conf_matrix)


# Classification Report

from sklearn.metrics import classification_report

report = classification_report(y_test.to_pandas(), y_pred)
print("Classification Report:\n", report)


# # Feature Importance: 
# import pandas as pd
# import numpy as np

# import pandas as pd
# import numpy as np

# feature_importance = np.abs(log_reg.coef_).flatten()
# features = X_train.columns  # No need to convert

# importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
# importance_df = importance_df.sort_values(by="Importance", ascending=False)

# print("Top Features Contributing to Predictions:")
# print(importance_df.head(10))
 # Import necessary libraries
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

# Convert cuDF to Pandas before prediction
X_test_pandas = X_test.to_pandas()
y_test_pandas = y_test.to_pandas()

# Predict on test data
y_pred = log_reg.predict(X_test_pandas)

# Accuracy Score
accuracy = accuracy_score(y_test_pandas, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_pandas, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
report = classification_report(y_test_pandas, y_pred)
print("Classification Report:\n", report)

# Feature Importance Calculation (Fixing length mismatch)
feature_importance = np.abs(log_reg.coef_).sum(axis=0)  # Sum across classes
features = X_train.columns  # No need to convert

# Create DataFrame for feature importance
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Display top 10 most important features
print("Top Features Contributing to Predictions:")
print(importance_df.head(10))


# Class Balancing with SMOTE (synthetic minority over-sampling technique)

from imblearn.over_sampling import SMOTE
import cupy as cp
import cudf

# Convert cuDF to NumPy for SMOTE
X_train_np = X_train.to_pandas().values
y_train_np = y_train.to_pandas().values

# Apply SMOTE to balance classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_np, y_train_np)

# Convert back to cuDF
X_train_balanced = cudf.DataFrame(X_resampled, columns=X_train.columns)
y_train_balanced = cudf.Series(y_resampled)

# Verify new class distribution
y_train_balanced_counts = y_train_balanced.value_counts()
y_train_balanced_counts



from imblearn.over_sampling import SMOTE
import cupy as cp
import cudf

# Convert cuDF to NumPy for SMOTE
X_train_np = X_train.to_pandas().values
y_train_np = y_train.to_pandas().values

# Apply SMOTE to balance classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_np, y_train_np)

# Convert back to cuDF
X_train_balanced = cudf.DataFrame(X_resampled, columns=X_train.columns)
y_train_balanced = cudf.Series(y_resampled)

# Verify new class distribution
y_train_balanced_counts = y_train_balanced.value_counts()
y_train_balanced_counts


import psutil
print(f"Available Memory: {psutil.virtual_memory().available / 1e9:.2f} GB")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gc
del X_train, y_train, X_test, y_test  # Delete large objects
gc.collect()  # Force garbage collection

from cuml.experimental.preprocessing import SMOTE
import cudf

# Apply GPU SMOTE
smote = SMOTE()
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(y_train_balanced.value_counts())


import joblib

# Save model coefficients and intercept
joblib.dump(log_reg, "/home/jmcphaul/WSL_Case Study 2//logistic_regression_model.pkl")

print("✅ Model saved successfully.")
 X_train.to_pandas().to_csv("/home/jmcphaul/WSL_Case Study 2/X_train_final.csv", index=False)
y_train.to_pandas().to_csv("/home/jmcphaul/WSL_Case Study 2/y_train_final.csv", index=False)
X_test.to_pandas().to_csv("/home/jmcphaul/WSL_Case Study 2/X_test_final.csv", index=False)
y_test.to_pandas().to_csv("/home/jmcphaul/WSL_Case Study 2/y_test_final.csv", index=False)

print("Final train/test data saved successfully.")
 

##############################################RUN FROM SAVED       ##############################################################################################################################################################


# run from saved;  
import joblib
import cudf
# Load trained model
log_reg = joblib.load("/home/jmcphaul/WSL_Case Study 2/logistic_regression_model.pkl")
# Load processed data
X_train = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/X_train_final.csv")
y_train = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/y_train_final.csv")
X_test = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/X_test_final.csv")
y_test = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/y_test_final.csv")

print("Reloaded saved model and data. Ready to continue.")



# CHECK MODEL PERFORMANCE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Make predictions
y_pred = log_reg.predict(X_test.to_pandas())
# Accuracy Score
accuracy = accuracy_score(y_test.to_pandas(), y_pred)
print(f"Accuracy: {accuracy:.4f}")
# Confusion Matrix
conf_matrix = confusion_matrix(y_test.to_pandas(), y_pred)
print("Confusion Matrix:\n", conf_matrix)
# Classification Report
report = classification_report(y_test.to_pandas(), y_pred)
print("Classification Report:\n", report)

###############################








## STILL TOO FEW  NEED MORE


iimport joblib
import cudf
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load saved data
X_train = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/X_train_final.csv")
y_train = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/y_train_final.csv")

# Convert to pandas for preprocessing
X_train_pd = X_train.to_pandas()
y_train_pd = y_train.to_pandas().values.ravel()  # Convert y to 1D array

# Apply StandardScaler to normalize feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_pd)

# Save the scaler for future use
joblib.dump(scaler, "/home/jmcphaul/WSL_Case Study 2/standard_scaler.pkl")

# Define improved parameter grid
param_grid = {
    "C": [0.1, 1.0],  
    "class_weight": ["balanced"],  
    "max_iter": [3000],  # Increased iterations for better convergence
    "solver": ["saga"],  # SAGA solver handles large datasets better
}

# Initialize GridSearch with optimized settings
grid_search = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=param_grid,
    scoring="accuracy",
    cv=2,  
    verbose=1,
    n_jobs=1  
)

# Run GridSearch on scaled data
grid_search.fit(X_train_scaled, y_train_pd)

# Get Best Parameters
print("Best Parameters Found:", grid_search.best_params_)

# Save Best Model
joblib.dump(grid_search.best_estimator_, "/home/jmcphaul/WSL_Case Study 2/best_logistic_regression.pkl")
print("Best model saved successfully.")









## HERE NEXT ##########################################

# Load best model and scaler
best_log_reg = joblib.load("/home/jmcphaul/WSL_Case Study 2/best_logistic_regression.pkl")
scaler = joblib.load("/home/jmcphaul/WSL_Case Study 2/standard_scaler.pkl")

# Load test data
X_test = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/X_test_final.csv")
y_test = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/y_test_final.csv")

# Scale test data
X_test_scaled = scaler.transform(X_test.to_pandas())

# Predict
y_pred_best = best_log_reg.predict(X_test_scaled)

# Accuracy Score
accuracy_best = accuracy_score(y_test.to_pandas(), y_pred_best)
print(f"Best Model Accuracy: {accuracy_best:.4f}")

# Confusion Matrix
conf_matrix_best = confusion_matrix(y_test.to_pandas(), y_pred_best)
print("Best Model Confusion Matrix:\n", conf_matrix_best)

# Classification Report
report_best = classification_report(y_test.to_pandas(), y_pred_best)
print("Best Model Classification Report:\n", report_best)














































































































































import joblib
import cudf
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
# Load saved model and data
log_reg = joblib.load("/home/jmcphaul/WSL_Case Study 2/logistic_regression_model.pkl")

X_train = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/X_train_final.csv")
y_train = cudf.read_csv("/home/jmcphaul/WSL_Case Study 2/y_train_final.csv")

# Sample 50% of the data to reduce memory usage
X_train_sample = X_train.to_pandas().sample(frac=0.5, random_state=42)
y_train_sample = y_train.to_pandas().sample(frac=0.5, random_state=42)
# Define reduced parameter grid
param_grid = {
    "C": [0.1, 1.0],  # Fewer values
    "class_weight": ["balanced"],  
    "max_iter": [200]  
}
# Initialize GridSearch with optimized settings
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring="accuracy",
    cv=2,  # Reduce CV folds
    verbose=1,
    n_jobs=1  # Run on a single core to prevent crashes
)
# Run GridSearch on sampled data
grid_search.fit(X_train_sample, y_train_sample)
# Get Best Parameters
print("Best Parameters Found:", grid_search.best_params_)
# Save Best Model
joblib.dump(grid_search.best_estimator_, "/home/jmcphaul/WSL_Case Study 2/best_logistic_regression.pkl")
print("Best model saved successfully.")
























































# crashing here: 

from sklearn.model_selection import GridSearchCV

# Reduce parameter combinations to avoid memory overload
param_grid = {
    "C": [0.1, 1.0],  # Fewer values
    "class_weight": ["balanced"],  # Use only balanced
    "max_iter": [200]  # Keep lower iteration limit
}

# Initialize GridSearch with optimized settings
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring="accuracy",
    cv=2,  # Reduce cross-validation folds to save memory
    verbose=1,
    n_jobs=2  # Reduce parallel jobs to avoid system overload
)

# Run GridSearch
grid_search.fit(X_train.to_pandas(), y_train.to_pandas())

# Get Best Parameters
print("Best Parameters Found:", grid_search.best_params_)

# Save Best Model
joblib.dump(grid_search.best_estimator_, "/home/jmcphaul/WSL_Case Study 2/best_logistic_regression.pkl")
print("Best model saved successfully.")











#now with metrics:

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Make predictions
y_pred = log_reg.predict(X_test.to_pandas())

# Accuracy Score
accuracy = accuracy_score(y_test.to_pandas(), y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test.to_pandas(), y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
report = classification_report(y_test.to_pandas(), y_pred)
print("Classification Report:\n", report)
 from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    "C": [0.01, 0.1, 1.0, 10],  # Regularization strength
    "class_weight": [{0: 1.0, 1: 1.5, 2: 3.0}, "balanced"],
    "max_iter": [200, 500],  # Increase iterations for better convergence
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,  # 3-Fold Cross Validation
    verbose=1,
    n_jobs=-1  # Use all available CPU cores
)

# Run GridSearch
grid_search.fit(X_train.to_pandas(), y_train.to_pandas())

# Get Best Parameters
print("Best Parameters Found:", grid_search.best_params_)

# Save Best Model
joblib.dump(grid_search.best_estimator_, "/home/jmcphaul/WSL_Case Study 2/best_logistic_regression.pkl")
print("Best model saved successfully.")


#

# Evaluate best model

# Load best model
best_log_reg = joblib.load("/home/jmcphaul/WSL_Case Study 2/best_logistic_regression.pkl")

# Make predictions
y_pred_best = best_log_reg.predict(X_test.to_pandas())

# Accuracy Score
accuracy_best = accuracy_score(y_test.to_pandas(), y_pred_best)
print(f"Best Model Accuracy: {accuracy_best:.4f}")

# Confusion Matrix
conf_matrix_best = confusion_matrix(y_test.to_pandas(), y_pred_best)
print("Best Model Confusion Matrix:\n", conf_matrix_best)

# Classification Report
report_best = classification_report(y_test.to_pandas(), y_pred_best)
print("Best Model Classification Report:\n", report_best)
 # Feat Importance & Viz
# Load best model
best_log_reg = joblib.load("/home/jmcphaul/WSL_Case Study 2/best_logistic_regression.pkl")

# Make predictions
y_pred_best = best_log_reg.predict(X_test.to_pandas())

# Accuracy Score
accuracy_best = accuracy_score(y_test.to_pandas(), y_pred_best)
print(f"Best Model Accuracy: {accuracy_best:.4f}")

# Confusion Matrix
conf_matrix_best = confusion_matrix(y_test.to_pandas(), y_pred_best)
print("Best Model Confusion Matrix:\n", conf_matrix_best)

# Classification Report
report_best = classification_report(y_test.to_pandas(), y_pred_best)
print("Best Model Classification Report:\n", report_best)




---#### 
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
###