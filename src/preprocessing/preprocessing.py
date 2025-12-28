import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE


##=================================================================================##
##                              Re Sampling
##=================================================================================##

def resampling(X, y, method='random'):
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
    elif method == 'random':
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
    elif method == 'smote':
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    elif method == 'adasyn':
        adasyn = ADASYN(random_state=42, n_neighbors=10)
        X_res, y_res = adasyn.fit_resample(X, y)
    elif method == 'borderline_smote':
        borderline_smote = BorderlineSMOTE(random_state=42)
        X_res, y_res = borderline_smote.fit_resample(X, y)
    else:
        raise ValueError("Unknown method: {}".format(method))
    return X_res, y_res

##=================================================================================##
##                              Preprocessing
##=================================================================================##

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

#=================================================================================##
##                      Highly Variable Genes Selection
##=================================================================================##

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

#=================================================================================##



