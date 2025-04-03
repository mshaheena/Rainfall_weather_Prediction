import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
import joblib

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sub = pd.read_csv("sample_submission.csv")

# Feature engineering
def enhance_features(df):
    df = df.copy()
    df['pressure_cut'] = df['pressure'] - 1000
    df['cloud_sunshine'] = df['cloud'] * df['sunshine'] / (np.sqrt(df['cloud']**2 + df['sunshine']**2) + 1e-8)
    df['humidity_dewpoint'] = df['humidity'] * df['sunshine'] / (np.sqrt(df['humidity']**2 + df['sunshine']**2) + 1e-8)
    
    df = df.rename(columns={
        'maxtemp': 'x1', 'temperature': 'x2', 'mintemp': 'x3', 'dewpoint': 'x4',
        'humidity': 'x5', 'cloud': 'x6', 'sunshine': 'x7', 'winddirection': 'x8',
        'windspeed': 'x9', 'pressure_cut': 'x10', 'cloud_sunshine': 'x11', 'humidity_dewpoint': 'x12'
    })
    
    df['_2_1'] = np.sqrt((df['x1'] - df['x3'])**2 + (df['x2'] - df['x4'])**2)
    df['_4_2'] = np.sqrt((df['x1'] - df['x9'])**2 + (df['x2'] - df['x10'])**2 +
                         (df['x3'] - df['x11'])**2 + (df['x4'] - df['x12'])**2)
    df['temp_range'] = df['x1'] - df['x3']
    df['humidity_pressure_ratio'] = df['x5'] / (df['x10'] + 0.0001)
    df['wind_cloud_interaction'] = df['x9'] * df['x6']
    df['temp_dewpoint_diff'] = df['x2'] - df['x4']
    df['sunshine_humidity_ratio'] = df['x7'] / (df['x5'] + 0.0001)
    return df

train = enhance_features(train)
test = enhance_features(test)

# Define features and target
features = [col for col in train.columns if col not in ['id', 'rainfall']]
X = train[features]
y = train['rainfall']
X_test = test[features]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
joblib.dump(imputer, 'imputer.pkl')

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

# Feature selection
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X_scaled, y)
X_test_selected = selector.transform(X_test_scaled)
joblib.dump(selector, 'selector.pkl')

# CatBoost Model
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
catboost_preds = np.zeros(len(test))
catboost_oof = np.zeros(len(train))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected, y)):
    X_tr, X_val = X_selected[train_idx], X_selected[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = CatBoostClassifier(iterations=500, learning_rate=0.03, depth=5, random_state=42, verbose=0, early_stopping_rounds=50)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
    catboost_oof[val_idx] = model.predict_proba(X_val)[:, 1]
    catboost_preds += model.predict_proba(X_test_selected)[:, 1] / kf.n_splits
    if fold == 0:
        model.save_model('catboost_model.cbm')

print(f"Overall CV AUC (CatBoost): {roc_auc_score(y, catboost_oof):.5f}")

# Logistic Regression Model
lr_preds = np.zeros(len(test))
lr_oof = np.zeros(len(train))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected, y)):
    X_tr, X_val = X_selected[train_idx], X_selected[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    lr_model = LogisticRegression(solver='liblinear', penalty='l1', C=0.05, random_state=42, max_iter=100)
    lr_model.fit(X_tr, y_tr)
    lr_oof[val_idx] = lr_model.predict_proba(X_val)[:, 1]
    lr_preds += lr_model.predict_proba(X_test_selected)[:, 1] / kf.n_splits
    if fold == 0:
        joblib.dump(lr_model, 'lr_model.pkl')

print(f"Overall CV AUC (Logistic Regression): {roc_auc_score(y, lr_oof):.5f}")

# Ensemble Predictions
final_preds = (catboost_preds + lr_preds) / 2
final_preds = np.clip(final_preds, 0.01, 0.99)

# Save submission
sub['rainfall'] = final_preds
sub.to_csv("submission.csv", index=False)
print("Submission file created!")
