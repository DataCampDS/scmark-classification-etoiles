"""
xgboost.py
Utility functions for training, tuning, and evaluating XGBoost models
for classification of 4 immune cell types:
Cancer_cells , NK_cells , T_cells_CD4+ , T_cells_CD8+
"""

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def labelencoding_for_xgboost(y_train, y_test) :
    le = LabelEncoder()
    y_enc_train = le.fit_transform(y_train)
    y_enc_test = le.transform(y_test)
    return (y_enc_train, y_enc_test)

# ------------------------------------------------------------
# Train a multi-class XGBoost model with regularization
# ------------------------------------------------------------
def train_xgboost(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,   # L1 regularization
    reg_lambda=1.0   # L2 regularization
):
    """
    Train a multi-class XGBoost classifier for 4 classes.
    """

    model = xgb.XGBClassifier(
        objective="multi:softprob", # can be "multi:softmax" if i want a class and not a prob for each class
        num_class=4,
        eval_metric="mlogloss", # or merror for accuracy
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        use_label_encoder=False,
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    return model, acc, report

# ------------------------------------------------------------
# Hyperparameter tuning for multi-class XGBoost
# ------------------------------------------------------------
def tune_xgboost(
    X_train,
    y_train,
    n_iter=50,
    cv=3,
    random_state=42
):
    """
    Randomized hyperparameter search for multi-class XGBoost.
    """

    param_dist = {
        "n_estimators": np.arange(100, 700, 50),
        "learning_rate": np.linspace(0.01, 0.3, 20),
        "max_depth": np.arange(3, 12),
        "subsample": np.linspace(0.6, 1.0, 10),
        "colsample_bytree": np.linspace(0.6, 1.0, 10),
        "reg_alpha": np.linspace(0, 5, 20),     # L1
        "reg_lambda": np.linspace(0.5, 5, 20),  # L2
        "gamma": np.linspace(0, 5, 20)
    }

    model = xgb.XGBClassifier(
        objective="multi:softprob", # can be "multi:softmax" if i want a class and not a prob for each class
        num_class=4,
        eval_metric="mlogloss", # or merror for accuracy
        use_label_encoder=False,
        tree_method="hist"
    )

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    return best_model, best_params


# ------------------------------------------------------------
# Evaluate a trained XGBoost model
# ------------------------------------------------------------
def evaluate_xgboost(model, X_test, y_test):
    """
    Computes accuracy and classification report for multi-class classification.
    """

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    return acc, report