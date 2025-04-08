# Load the necessary libraries
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay  
from xgboost import XGBClassifier




# Establish a connection to the SQLite database (or create it if it doesn't exist)
con = sqlite3.connect("RealFakeJobs.db")
# Create a cursor object to interact with the database
cur = con.cursor()




# Define the query to select variables for the analysis
query = '''
    SELECT 
        Title,  
        Company_profile,
        Description,
        Requirements,
        Benefits,
        Telecommuting,
        Logo,
        Has_questions,
        Employment_type,
        Required_experience,
        Required_education,
        Industry,
        Job_function,
        Fraudulent,
        State,
        City
    FROM Job_Posts_US;
'''

# Execute and load into DataFrame
cur.execute(query)
columns = [desc[0] for desc in cur.description]
df_full = pd.DataFrame(cur.fetchall(), columns=columns)

# Preview
df_full.head()


# Convert 'unspecified' entries to 0, others to 1
for col in ['Company_profile', 'Description', 'Requirements', 'Benefits']:
    df_full[col] = df_full[col].apply(lambda x: 0 if x.strip().lower() == 'unspecified' else 1)


# Define features and target
features = [
    'Title', 'Company_profile', 'Description', 'Requirements', 'Benefits', 'Telecommuting', 'Logo', 'Has_questions',
    'Employment_type', 'Required_experience', 'Required_education', 'Industry', 'Job_function', 'State', 'City']

X = df_full[features]
y = df_full['Fraudulent']


# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)


# Split Data into Training and Test Sets (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y) 




# Calculate class weight (inverse ratio of class distribution)
ratio = y_train.value_counts()[0] / y_train.value_counts()[1]  # (Real / Fake)


xgb_model = XGBClassifier(
    eval_metric="logloss",     
    n_estimators=150,
    max_depth=5,               # reduce tree depth to prevent overfitting
    scale_pos_weight=ratio,    # handle class imbalance
    use_label_encoder=False,
    random_state=42
)

# Fit the model
xgb_model.fit(X_train, y_train)


# Make Predictions on Training and Test Data
y_train_pred_xgb = xgb_model.predict(X_train)
y_test_pred_xgb = xgb_model.predict(X_test)

# Compute Accuracy for Training and Test Sets
xgb_train_accuracy = accuracy_score(y_train, y_train_pred_xgb)
xgb_test_accuracy = accuracy_score(y_test, y_test_pred_xgb)

# Print Results
print(f"XGBoost Training Accuracy: {xgb_train_accuracy:.4f}")
print(f"XGBoost Test Accuracy: {xgb_test_accuracy:.4f}")


# Get probability scores 
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]


# Compute Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_test_pred_xgb)

# Display Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 3))  
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=["Real", "Fake"])
disp_xgb.plot(cmap="Blues", ax=ax)  
plt.title("Confusion Matrix - XGBoost")
plt.show()


# Compute ROC values
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
auc_score = roc_auc_score(y_test, y_prob_xgb)

# Plot ROC
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc_score:.2f})', color='darkred', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Get feature importances and sort them in descending order
sorted_features_xgb = sorted(zip(X_encoded.columns, xgb_model.feature_importances_), key=lambda x: x[1], reverse=True)

print("XGBoost Feature Importances:")
for feat, imp in sorted_features_xgb[:10]:
    print(feat, imp)