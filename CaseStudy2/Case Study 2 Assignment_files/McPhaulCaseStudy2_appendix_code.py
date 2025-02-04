!pip install cupy-cuda11x
!pip install cudf-cu11
!pip install imblearn

from google.colab import drive
import psutil
import torch
import cupy as cp
import torch
import cudf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import numpy as np
from sklearn.metrics import roc_curve, auc
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import locale
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize


# Mount Google Drive (if needed)
from google.colab import drive
drive.mount('/content/drive')


# Load the datasets
diabetic_data = pd.read_csv("/content/drive/MyDrive/diabetic_data.csv")
ids_mapping = pd.read_csv("/content/drive/MyDrive/IDs_mapping.csv")

# Data Cleaning and Merging
# Convert 'admission_type_id' to numeric, handling non-numeric values
diabetic_data['admission_type_id'] = pd.to_numeric(diabetic_data['admission_type_id'], errors='coerce')
ids_mapping['admission_type_id'] = pd.to_numeric(ids_mapping['admission_type_id'], errors='coerce')

# Convert to Int64 after ensuring both are numeric
diabetic_data['admission_type_id'] = diabetic_data['admission_type_id'].astype('Int64')
ids_mapping['admission_type_id'] = ids_mapping['admission_type_id'].astype('Int64')

# Merge diabetic_data with ids_mapping
diabetic_data = diabetic_data.merge(ids_mapping, how="left", on="admission_type_id")

# Fill missing values in key categorical columns
for col in ["race", "diag_1", "diag_2", "diag_3"]:
    diabetic_data[col] = diabetic_data[col].fillna("Unknown")

# Convert 'readmitted' to numerical categories
diabetic_data["readmitted"] = diabetic_data["readmitted"].map({"NO": 0, ">30": 1, "<30": 2})

# Convert 'max_glu_serum' and 'A1Cresult' to numerical representations
diabetic_data['max_glu_serum'] = diabetic_data['max_glu_serum'].replace({
    'None': 0,
    'Norm': 1,
    '>200': 2,
    '>300': 3
})

diabetic_data['A1Cresult'] = diabetic_data['A1Cresult'].replace({
    'None': 0,
    'Norm': 1,
    '>7': 2,
    '>8': 3
})

# Drop unnecessary columns
columns_to_drop = [
    "weight", "medical_specialty", "payer_code",
    "encounter_id", "patient_nbr", "description"
]
diabetic_data = diabetic_data.drop(columns=columns_to_drop, errors='ignore')

# Feature Engineering (One-Hot Encoding)
categorical_cols = ["race", "gender", "age", "change", "diabetesMed", "insulin"]
diabetic_data = pd.get_dummies(diabetic_data, columns=categorical_cols, dummy_na=True)

# Feature Engineering (Convert to Numeric)
for col in ['diag_1', 'diag_2', 'diag_3']:
    diabetic_data[col] = diabetic_data[col].astype('category').cat.codes

medication_cols = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
]
for col in medication_cols:
    if diabetic_data[col].dtype == 'object':
        diabetic_data[col] = (diabetic_data[col].astype(str) != "No").astype("int32")

# Convert to float32
diabetic_data = diabetic_data.astype("float32")

# Split Data
X = diabetic_data.drop(columns=['readmitted'])
y = diabetic_data['readmitted'].astype("int32")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Initialize and Train Logistic Regression Model
best_log_reg = LogisticRegression(C=1.0, class_weight={0: 1.0, 1: 2.0, 2: 4.0}, max_iter=3000, solver='saga')
best_log_reg.fit(X_resampled, y_resampled)


# Save the best model to a file
joblib.dump(best_log_reg, "bestModel.pkl")
print("Model saved as bestModel.pkl")

# To load the model later:
# loaded_model = joblib.load("bestModel.pkl")
# y_pred_loaded = loaded_model.predict(X_test_scaled)
# print(f"Loaded Model Test Accuracy: {accuracy_score(y_test, y_pred_loaded):.4f}")


# Predict on Test Data
y_pred = best_log_reg.predict(X_test_scaled)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)


# ROC Curve and AUC (Multi-class)
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

n_classes = len(np.unique(y_test))
y_test_bin = label_binarize(y_test, classes=np.unique(y_test)) # Binarize the output
fpr = dict()
tpr = dict()
roc_auc = dict()

y_pred_proba = best_log_reg.predict_proba(X_test_scaled)


for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
  plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--') # Random classifier line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for Multi-Class')
plt.legend(loc="lower right")
plt.show()


# Feature Importance (Top 5)
feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': abs(best_log_reg.coef_[0])})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
print("\nTop 5 Important Features:")
print(feature_importances.head(5))


# Final Submission Check
# osave predictions to a CSV file

submission_df = pd.DataFrame({'prediction': y_pred})
submission_df.to_csv('final_predictions.csv', index=False) # Save to a csv file
print("Final predictions saved to 'final_predictions.csv'")


# compile case study from daibetes analysis 
# Data Exploration
print("\nData Exploration:")
print(df.describe())  # Summary statistics
# Feature Engineering (if applicable)
print("\nFeature Engineering:")
# Model Comparison (if you've tried other models)
print("\nModel Comparison:")
# Hyperparameter Tuning for other models
print("\nHyperparameter Tuning:")
# Conclusion
print("\nConclusion:")

