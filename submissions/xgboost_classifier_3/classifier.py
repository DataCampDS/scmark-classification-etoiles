import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

def xgboost_pipeline(scaler: bool, pca_n_components, max_depths, max_features):
    """
    Docstring for xgboost_pipeline
    
    :param scaler: Whether to include a StandardScaler in the pipeline
    :param n_components: Number of components for PCA
    :param max_depths: Maximum depth for the HistGradientBoostingClassifier
    :param max_features: Maximum features for the HistGradientBoostingClassifier
    """
    
    pipe = Pipeline([])
    if scaler:
        pipe.steps.append(
            ("Scaler", StandardScaler(with_mean=True, with_std=True)),
        )
    
    pipe.steps.append(
        ("PCA with 50 components", PCA(n_components=pca_n_components)),
    )

    pipe.steps.append(
        ("XBBoost Classifier",
            HistGradientBoostingClassifier(max_depth=max_depths),
        ),
    )

    return pipe

def preprocess(X, method='none'):
    """
    Convertit la matrice sparse en dense
    Normalisation : chaque cellule a une somme = 1
    """
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    if method == 'none':
        pass
    elif method == 'normalize':
        X = X / X.sum(axis=1)[:, np.newaxis]
    elif method == 'log':
        X = np.log1p(X)
    else:
        raise ValueError("Unknown preprocessing method: {}".format(method))
    return X

def resampling(X, y, method='none'):
    """
    Resampling des données pour gérer le déséquilibre des classes
    Méthodes disponibles :
    - 'none' : Pas de resampling
    - 'random' : RandomOverSampler
    - 'smote' : SMOTE
    - 'adasyn' : ADASYN
    - 'borderline_smote' : BorderlineSMOTE
    """
    if method == 'none':
        return X, y
    return X_res, y_res

def select_HVG(X, N_HVG):
    """
    Sélection des N_HVG gènes les plus variables
    """
    if N_HVG is None:
        return X
    
    gene_vars = X.var(axis=0)
    hvg_idx = np.argsort(gene_vars)[-N_HVG:]

    X_hvg_selected = X[:, hvg_idx]
    return X_hvg_selected

class Classifier(object):
    def __init__(self):
        # Use scikit-learn's pipeline
        self.pipe = xgboost_pipeline(
            scaler=False,
            pca_n_components=30,
            max_depths=7,
            max_features=0.7
        )

    def fit(self, X_sparse, y):
        X_res, y_res = resampling(X_sparse, y, method='none')
        X_res = preprocess(X_res, method='log')
        self.pipe.fit(X_res, y_res)

        pass

    def predict_proba(self, X_sparse):
        X = preprocess(X_sparse, method='log')
        X = select_HVG(X, N_HVG=None)
        # here we use HistGradientBoosting.predict_proba()
        return self.pipe.predict_proba(X)
