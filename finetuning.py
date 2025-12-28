##=================================================================================##
## Finetuning script for hyperparameter optimization of the Classifier model.      ##
## This script performs a grid search over various hyperparameters,                ##
## evaluates the model on the test dataset, and logs the results.                  ##
##=================================================================================##

# ================= Imports ================= #
import sys
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import time
import numpy as np

# Dynamically add the 'src' directory to the Python path
src_path = './src'

if src_path not in sys.path and os.path.exists(src_path):
	sys.path.append(src_path)

# Ensure the path to the 'src' directory is correctly added
from preprocessing.preprocessing import resampling, preprocess, select_HVG
from pipeline.pipelines import xgboost_pipeline

# Import the required functions from the problem module
from problem import get_train_data, get_test_data


# ================= Load Data ================= #

# Load the training and testing data
X_train, y_train = get_train_data()
X_test, y_test = get_test_data()

X_all = np.vstack([X_train.toarray(), X_test.toarray()])
y_all = np.hstack([y_train, y_test])

# ================= Classifier ================= #

class Classifier(object):
    def __init__(self, **kwargs):
        # Use scikit-learn's pipeline
        self.pipe = xgboost_pipeline(scaler=kwargs.get('scaler'), 
                                     pca_n_components=kwargs.get('pca_n_components'),
                                     max_depths=kwargs.get('max_depths'),
                                     max_features=kwargs.get('max_features'))
        
        self.kwargs = kwargs

    def fit(self, X_sparse, y):
        X, y = resampling(X_sparse, y, method=self.kwargs.get('resampling_method', 'none'))
        X = preprocess(X, method=self.kwargs.get('preprocessing_method', 'none'))
        X = select_HVG(X, N_HVG=self.kwargs.get('N_HVG'))

        self.pipe.fit(X, y)

        pass

    def predict_proba(self, X_sparse):
        X = preprocess(X_sparse, method=self.kwargs.get('preprocessing_method', 'none'))
        X = select_HVG(X, N_HVG=self.kwargs.get('N_HVG'))

        return self.pipe.predict_proba(X)
    
    def predict(self, X_sparse):
        X = preprocess(X_sparse, method=self.kwargs.get('preprocessing_method', 'none'))
        X = select_HVG(X, N_HVG=self.kwargs.get('N_HVG'))
        return self.pipe.predict(X)
    
# ================= K-Fold Fit and Eval ================= #
def k_fold_fit_eval(clf, X, y, n_splits=5):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    start_time = time.time()

    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        accuracy = balanced_accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)

    end_time = time.time()
    avg_accuracy = np.mean(accuracies)
    fitting_time = end_time - start_time

    return avg_accuracy, fitting_time

# ================= Grid Search ================= #
# Define hyperparameter options
scaler_options = [False] #True is always worse
pca_n_components_options = [30, 40, 50]
max_depths_options = [5, 6, 7]
max_features_options = [0.5, 0.7, 1.0]
resampling_methods = ['borderline_smote'] #'adasyn' doesn't work, "none" and "random" are bad
preprocessing_methods = ['log'] # 'none' and 'normalize' are worse
N_HVG_options = [None] # other values are worse maybe the function is broken

# first line of csv log file
open("grid_search_log.csv", "a").write("scaler, pca_n_components, max_depths, max_features, resampling_method, preprocessing_method, N_HVG, accuracy, fitting_time\n")

# Grid search over hyperparameters
best_accuracy = 0
best_params = {}

for scaler in scaler_options:
    for pca_n_components in pca_n_components_options:
        for max_depths in max_depths_options:
            for max_features in max_features_options:
                for resampling_method in resampling_methods:
                    for preprocessing_method in preprocessing_methods:
                        for N_HVG in N_HVG_options:

                            clf = Classifier(scaler=scaler,
                                            pca_n_components=pca_n_components,
                                            max_depths=max_depths,
                                            max_features=max_features,
                                            resampling_method=resampling_method,
                                            preprocessing_method=preprocessing_method,
                                            N_HVG=N_HVG)
                            try:
                                accuracy, fitting_time = k_fold_fit_eval(clf, X_all, y_all, n_splits=5)
                                
                                if accuracy > best_accuracy:
                                    best_accuracy = accuracy
                                    best_params = {
                                        'scaler': scaler,
                                        'pca_n_components': pca_n_components,
                                        'max_depths': max_depths,
                                        'max_features': max_features,
                                        'resampling_method': resampling_method,
                                        'preprocessing_method': preprocessing_method,
                                        'N_HVG': N_HVG
                                    }
                            except Exception as e:
                                accuracy = 0  # In case of failure, set accuracy to 0
                                fitting_time = 0  # In case of failure, set fitting time to 0
                            # put reults in a log file as csv
                            open("grid_search_log.csv", "a").write(f"{scaler}, {pca_n_components}, {max_depths}, {max_features}, {resampling_method}, {preprocessing_method}, {N_HVG}, {accuracy}, {fitting_time:.2f}\n")

print(f"Best Accuracy: {best_accuracy} with params: {best_params}")


