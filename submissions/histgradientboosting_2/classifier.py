from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class HVGSelector:
    def __init__(self, n_hvg=2000):
        self.n_hvg = n_hvg

    def fit(self, X):
        mean = X.mean(axis=0)
        var = X.var(axis=0)

        dispersion = var / (mean + 1e-6)
        self.idx_ = np.argsort(dispersion)[-self.n_hvg:]
        return self

    def transform(self, X):
        return X[:, self.idx_]


class Classifier:
    def __init__(self):
        self.hvg = HVGSelector(n_hvg=1000)

        self.model = HistGradientBoostingClassifier(
            
            learning_rate=0.05,
            max_depth=7,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=0.5,
            early_stopping=True,
            random_state=0
        )

        

    def preprocess(self,X):
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # library size normalization
        libsize = X.sum(axis=1, keepdims=True)
        libsize[libsize == 0] = 1.0
        X = X / libsize * 1e4

        # log transform
        X = np.log1p(X)
        return X

    def fit(self, X_sparse, y):
        X = self.preprocess(X_sparse)

        # HVG selection (train only)
        self.hvg.fit(X)
        X = self.hvg.transform(X)

        # Handle class imbalance
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y
        )

        self.model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X_sparse):
        X = self.preprocess(X_sparse)
        X = self.hvg.transform(X)
        return self.model.predict_proba(X)
