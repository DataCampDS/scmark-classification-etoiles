from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier


##=================================================================================##
##                              XGBoost Pipeline
##=================================================================================##

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
            HistGradientBoostingClassifier(max_depth=max_depths, max_features=max_features),
        ),
    )

    return pipe

##=================================================================================##

