import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


class SARIMAXForecaster:
    def __init__(
        self,
        order=(2, 0, 1),
        trend=None,
        freq="MS",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ):
        self.order = order
        self.trend = trend
        self.freq = freq
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.res_ = None
        self.k_exog_ = None
        self.y_train_ = None

    def fit(self, y_train, X_train_exog=None):
        if X_train_exog is None:
            exog = None
            self.k_exog_ = 0
        else:
            exog = pd.DataFrame(X_train_exog, index=y_train.index)
            self.k_exog_ = exog.shape[1]

        try:
            mod = SARIMAX(
                endog=y_train,
                exog=exog,
                order=self.order,
                seasonal_order=(0, 0, 0, 0),
                trend=self.trend,
                freq=self.freq,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
            )
            self.res_ = mod.fit(disp=False, maxiter=500)
            self.y_train_ = y_train.copy()
        except Exception as ex:
            print(f"Erreur SARIMAX à {ex}")
            self.res_ = None
            self.y_train_ = y_train.copy()
        return self

    def predict_h(self, X_future_exog=None, h=1):
        if self.res_ is None:
            return np.repeat(self.y_train_.iloc[-1], h)

        if self.k_exog_ == 0:
            exog_f = None
        else:
            exog_f = pd.DataFrame(
                X_future_exog,
                index=pd.date_range(
                    start=self.y_train_.index[-1], periods=h + 1, freq=self.freq
                )[1:],
            )
            assert exog_f.shape[0] == h, f"exog futur shape {exog_f.shape[0]} vs h={h}"

        try:
            fc = self.res_.get_forecast(steps=h, exog=exog_f)
            return fc.predicted_mean.values
        except Exception as ex:
            print(f"Erreur SARIMAX à h={h} : {ex}")
            return np.repeat(self.y_train_.iloc[-1], h)


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
