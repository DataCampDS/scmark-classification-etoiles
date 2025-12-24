import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline


def preprocess_log(X):
    """
    Application of log(1 + x) to reduce the gap between the values
    """
    X = X.toarray()  # Convert sparse matrix to dense
    X_log = np.log1p(X)
    return X_log

class Classifier(object):
    def __init__(self):
        # Use scikit-learn's pipeline
        self.pipe = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=50),
            HistGradientBoostingClassifier(
                max_depth=2
            ),
        )

    def fit(self, X_sparse, y):
        X = preprocess_log(X_sparse)
        self.pipe.fit(X, y)

        pass

    def predict_proba(self, X_sparse):
        X = preprocess_log(X_sparse)
        # here we use HistGradientBoosting.predict_proba()
        return self.pipe.predict_proba(X)
