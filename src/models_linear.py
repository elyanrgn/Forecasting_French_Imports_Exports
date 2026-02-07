import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


def fit_ridge_ts_cv(X_train, y_train, n_splits=3, alphas=None):
    if alphas is None:
        alphas = np.logspace(-6, 6, 25)

    pipe = Pipeline(
        [("scaler", StandardScaler(with_mean=True, with_std=True)), ("model", Ridge())]
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    gs = GridSearchCV(
        pipe,
        param_grid={"model__alpha": alphas},
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def fit_lasso_ts_cv(X_train, y_train, n_splits=3, n_alphas=100, max_iter=200000):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MultiTaskLassoCV(
                    cv=tscv, n_alphas=n_alphas, max_iter=max_iter, random_state=42
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def fit_elasticnet_ts_cv(
    X_train, y_train, n_splits=3, l1_ratios=(0.1, 0.5, 0.9), max_iter=200000
):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MultiTaskElasticNetCV(
                    l1_ratio=list(l1_ratios),
                    cv=tscv,
                    max_iter=max_iter,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model
