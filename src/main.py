import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

# 1. LOAD DATA 
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("── Raw Data ──")
print(df.shape)
print(df.head())
print(df.info())

# 2. CLEAN DATA

# Drop customerID - random ID, useless for prediction
df = df.drop(columns=['customerID'])

# TotalCharges is stored as text - convert to number
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill the blanks created above with the median value
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Convert target column: Yes=1, No=0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Convert all remaining text columns to numbers
df = pd.get_dummies(df, drop_first=True)

print("\n── After Cleaning ──")
print(f"Missing values: {df.isnull().sum().sum()}")  # should be 0
print(f"Shape: {df.shape}")                          # should be (7043, 31)

# 3. SPLIT DATA

# X = inputs (everything except Churn)
# y = target (Churn column only)
X = df.drop('Churn', axis=1)
y = df['Churn']

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. TRAIN & COMPARE MODELS
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000),
    'Random Forest':       RandomForestClassifier(n_estimators=100),
    'XGBoost':             XGBClassifier(eval_metric='logloss')
}

print("\n── Model Results ──")
for name, model in models.items():
    # Train on 80% of data
    model.fit(X_train, y_train)

    # Predict on the 20% the model has never seen
    preds = model.predict(X_test)

    # Print results
    print(f"\n--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_test, preds):.3f}")
    print(f"F1 Score : {f1_score(y_test, preds):.3f}")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))