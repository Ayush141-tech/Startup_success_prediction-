import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set seed so we get same random numbers
np.random.seed(42)

# Step 1: Create dataset (200 startups)
n = 200

funding = np.random.uniform(0.5, 10.0, n)        # Funding in Millions
revenue = np.random.uniform(0.1, 8.0, n)         # Revenue in Millions
team_size = np.random.randint(3, 50, n)          # Number of employees
valuation = funding * np.random.uniform(2, 5, n)   # Valuation in Millions

# Success score formula with simple noise
score = (revenue * 2) + (valuation / funding) + (team_size * 0.05) + np.random.normal(0, 1.2, n)
success = (score > np.median(score)).astype(int)

# Create dataframe
df = pd.DataFrame({
    'Funding': np.round(funding, 2),
    'Revenue': np.round(revenue, 2),
    'Team_Size': team_size,
    'Valuation': np.round(valuation, 2),
    'Success': success
})

print("Dataset Preview:")
print(df.head())
print()

# Step 2: Separate features and target
X = df[['Funding', 'Revenue', 'Team_Size', 'Valuation']]
y = df['Success']

# Step 3: Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------
# Model 1: Logistic Regression
# ------------------------------------
lr = LogisticRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred)
lr_rec = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print("Logistic Regression:")
print("Accuracy :", round(lr_acc, 4))
print("Precision:", round(lr_prec, 4))
print("Recall   :", round(lr_rec, 4))
print("F1 Score :", round(lr_f1, 4))
print()

# ------------------------------------
# Model 2: Decision Tree
# ------------------------------------
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
dt_prec = precision_score(y_test, dt_pred)
dt_rec = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

print("Decision Tree:")
print("Accuracy :", round(dt_acc, 4))
print("Precision:", round(dt_prec, 4))
print("Recall   :", round(dt_rec, 4))
print("F1 Score :", round(dt_f1, 4))
print()

# ------------------------------------
# Model 3: Random Forest
# ------------------------------------
rf = RandomForestClassifier(n_estimators=40, max_depth=3, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred)
rf_rec = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("Random Forest:")
print("Accuracy :", round(rf_acc, 4))
print("Precision:", round(rf_prec, 4))
print("Recall   :", round(rf_rec, 4))
print("F1 Score :", round(rf_f1, 4))
print()

# ------------------------------------
# Model 4: XGBoost
# ------------------------------------
xgb = XGBClassifier(n_estimators=25, max_depth=2, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_prec = precision_score(y_test, xgb_pred)
xgb_rec = recall_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)

print("XGBoost:")
print("Accuracy :", round(xgb_acc, 4))
print("Precision:", round(xgb_prec, 4))
print("Recall   :", round(xgb_rec, 4))
print("F1 Score :", round(xgb_f1, 4))
print()

# ------------------------------------
# Summary Table
# ------------------------------------
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'Accuracy': [round(lr_acc, 4), round(dt_acc, 4), round(rf_acc, 4), round(xgb_acc, 4)],
    'Precision': [round(lr_prec, 4), round(dt_prec, 4), round(rf_prec, 4), round(xgb_prec, 4)],
    'Recall': [round(lr_rec, 4), round(dt_rec, 4), round(rf_rec, 4), round(xgb_rec, 4)],
    'F1 Score': [round(lr_f1, 4), round(dt_f1, 4), round(rf_f1, 4), round(xgb_f1, 4)]
})

print("=== Final Comparison Table ===")
print(results)
print()

# Test prediction for a new startup
new_startup = pd.DataFrame({
    'Funding': [5.0],
    'Revenue': [3.2],
    'Team_Size': [20],
    'Valuation': [22.0]
})

pred = rf.predict(new_startup)[0]

print("Prediction for new startup:")
print(new_startup)

if pred == 1:
    print("Result: 1 (Startup is likely to succeed)")
else:
    print("Result: 0 (Startup is likely to fail)")
