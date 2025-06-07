# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

def train_random_forest(data, features, target):
    koin_models = {}
    for coin in data['coin'].unique():
        df_coin = data[data['coin'] == coin]
        X = df_coin[features]
        y = df_coin[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        koin_models[coin] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
    return koin_models

def train_svm(data, features, target):
    koin_models = {}
    for coin in data['coin'].unique():
        df_coin = data[data['coin'] == coin]
        X = df_coin[features]
        y = df_coin[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = SVC(random_state=42, probability=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        koin_models[coin] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
    return koin_models

def train_xgboost(data, features, target):
    koin_models = {}
    for coin in data['coin'].unique():
        df_coin = data[data['coin'] == coin]
        X = df_coin[features]
        y = df_coin[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        koin_models[coin] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
    return koin_models
