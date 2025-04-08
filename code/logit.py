# Load the necessary libraries
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay  

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


# Train a logistic regression model with L1 regularization
model_l1 = LogisticRegression(
    class_weight="balanced",  # Adjusts for class imbalance 
    penalty='l1',             # Applies L1 regularization (lasso), which can zero out less important features
    C=0.5,                    # Inverse of regularization strength; smaller values = stronger regularization
    solver='liblinear',       # Optimization algorithm that supports L1 regularization (good for medium datasets)
    random_state=42           
)
model_l1.fit(X_train, y_train)


# Predict on test data
y_pred = model_l1.predict(X_test)

# Get probabilities
y_prob = model_l1.predict_proba(X_test)[:, 1]

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")


# Print classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


# Compute Confusion Matrix
cm_blr1 = confusion_matrix(y_test, y_pred)

# Display Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 3))  
disp = ConfusionMatrixDisplay(confusion_matrix=cm_blr1, display_labels=["Real", "Fake"])
disp.plot(cmap="Blues", ax=ax)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()


# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic regression with L1 regularization')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()


# Get model coefficients
coefs = model_l1.coef_[0]  # Flatten the array (since it's binary classification)

# Count non-zero and zero coefficients
num_nonzero = np.sum(coefs != 0)
num_zero = np.sum(coefs == 0)

print(f"Total features: {len(coefs)}")
print(f"Non-zero coefficients (selected features): {num_nonzero}")
print(f"Zero coefficients (shrunk to zero): {num_zero}")


# Match coefficients to feature names
feature_names = X_train.columns
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})

# Sort by absolute value (strongest impact)
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df_sorted = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

print(coef_df_sorted[['Feature', 'Coefficient']].head(10))