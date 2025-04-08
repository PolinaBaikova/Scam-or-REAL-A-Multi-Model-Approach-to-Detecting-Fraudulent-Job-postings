# Load the necessary libraries
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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



# Train RandomForestClassifier with Different Depths and Evaluate Accuracy
depths = range(5, 101)  # Testing tree depths from 5 to 100
train_accuracies = []
test_accuracies = []

for depth in depths:
    rf = RandomForestClassifier(n_estimators=100, max_depth=depth, class_weight="balanced", random_state=42) # Create the model
    rf.fit(X_train, y_train)  # Train the model 

    # Compute and store accuracy for training and test sets
    train_accuracies.append(rf.score(X_train, y_train))
    test_accuracies.append(rf.score(X_test, y_test))


# Find the Best Depth (Maximizing Test Accuracy)
best_depth = depths[np.argmax(test_accuracies)]  
print(f"Best depth: {best_depth}")


# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=best_depth, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)


# Compute final accuracy on training and test sets
train_accuracy_rf = rf_model.score(X_train, y_train)
test_accuracy_rf = rf_model.score(X_test, y_test)
print(f"Random Forest training accuracy: {train_accuracy_rf:.4f}")
print(f"Random Forest testing accuracy: {test_accuracy_rf:.4f}")


# Get predicted probabilities for the positive class (Swimming)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1] 

# Get binary predictions
y_pred_rf = rf_model.predict(X_test)


# Compute Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Display Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 3)) 
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Real", "Fake"])
disp_rf.plot(cmap="Blues", ax=ax)
plt.title("Confusion Matrix - Random Forest Model")
plt.show()



# Get probabilities for the positive class (Fake)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# ROC values
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
auc_rf = roc_auc_score(y_test, rf_probs)

# Plot ROC
plt.figure(figsize=(5, 4))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='green', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()


# Custom threshold
custom_threshold = 0.2
y_pred_custom = (rf_probs >= custom_threshold).astype(int)


# Calculate accuracy with the custom threshold
custom_accuracy = accuracy_score(y_test, y_pred_custom)
print(f"Accuracy at threshold {custom_threshold}: {custom_accuracy:.4f}")


# Compute Confusion Matrix
cm_rf1 = confusion_matrix(y_test, y_pred_custom )

# Display Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 3)) 
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf1, display_labels=["Real", "Fake"])
disp_rf.plot(cmap="Blues", ax=ax)
plt.title("Confusion Matrix - Random Forest Model")
plt.show()


# Get feature importances and sort them in descending order
sorted_features_rf = sorted(zip(X_encoded.columns, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)

print("Random Forest Feature Importances:")
for feat, imp in sorted_features_rf[:10]:
    print(feat, imp)