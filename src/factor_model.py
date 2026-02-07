import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


def fit_dfm_pca_ridge_ts_cv(
    X_train, y_train, n_splits=3, r_grid=(2, 4, 6, 8), ridge_alphas=None
):
    if ridge_alphas is None:
        ridge_alphas = np.logspace(-6, 6, 15)

    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA()), ("model", Ridge())])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    gs = GridSearchCV(
        pipe,
        param_grid={"pca__n_components": list(r_grid), "model__alpha": ridge_alphas},
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_
