import numpy as np 
import pandas as pd 
from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data[pd.get_dummies(train_data[['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']]).columns] = pd.get_dummies(train_data[['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']])
train_data_1 = train_data.drop(['id','person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], axis = 1)
test_data[pd.get_dummies(test_data[['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']]).columns] = pd.get_dummies(test_data[['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']])
test_data_1 = test_data.drop(['id','person_home_ownership','loan_intent', 'loan_grade', 'cb_person_default_on_file'], axis = 1)

y_train = train_data['loan_status']
train_data_1.drop('loan_status', axis=1, inplace=True)
train_data_1.replace({True:1, False:0}, inplace=True)
test_data_1.replace({True:1, False:0}, inplace=True)
train_data_numpy = train_data_1.to_numpy()
test_data_numpy = test_data_1.to_numpy()
y_train_numpy = y_train.to_numpy()

X_train, X_val, y_train, y_val = train_test_split(train_data_numpy, y_train_numpy, test_size=0.2, random_state=42, shuffle=True)

def objective(trial):
    param = {
        'max_depth': 6,
        'n_estimators': trial.suggest_int('n_estimators', 80, 300), 
        'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.3),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'eta':trial.suggest_float('eta', 0.01, 10),
        'reg_lambda':trial.suggest_float('reg_lambda', 30, 100),
        'reg_alpha':trial.suggest_float('reg_alpha', 1, 10),
        'max_leaves' : trial.suggest_int('max_leaves', 1, 10),
        'eval_metric' : 'logloss'
    }
    
    model = XGBClassifier(**param, device='cuda')
    model.fit(X_train, y_train, verbose=False)
    y_predict = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_predict)
    return accuracy

study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 5000)

xgb_model = XGBClassifier(**study.best_params)
xgb_model.fit(train_data_numpy, y_train_numpy)

y_pred = xgb_model.predict(test_data_numpy)
out = pd.DataFrame({'id': test_data['id'], 'loan_status': y_pred})

out.to_csv('out.csv', index=False)
