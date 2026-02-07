from models_linear import (
    SARIMAXForecaster,
    fit_ridge_ts_cv,
    fit_lasso_ts_cv,
    fit_elasticnet_ts_cv,
    fit_dfm_pca_ridge_ts_cv,
)
import pandas as pd
import numpy as np
from tqdm import tqdm


def rolling_window_complete(
    X,
    y,
    rf_model,
    xgb_model,
    nn_model,
    horizons=(1, 3, 6, 12),
    start_test=None,
    comparison_length=48,
    fixed_train_window=None,
    verbose=True,
):
    """
    Rolling window complet:
    - X: DataFrame (T, n_feat), index datetime
    - y: DataFrame (T, 2), index datetime
    - horizons: tuple, ex (1,3,6,12)
    - comparison_length: int, ex 48 (fenêtres glissantes pour calcul wins)
    """
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame)
    # X, y = X.align(y, join="inner", axis=0)
    assert X.index.equals(y.index)

    if start_test is None:
        start_idx = int(len(X) * 0.7)
    else:
        start_idx = X.index.get_loc(start_test)

    if verbose:
        print(
            f"Rolling window: test start={X.index[start_idx]}, "
            f"horizons={horizons}, comparison_length={comparison_length}\n"
        )

    # noms modèles
    ml_names = ["RF", "XGB", "NN"]
    linear_names = ["Ridge", "Lasso", "ElasticNet", "DFM"]
    sarimax_names = [
        "SARIMAX_y1_noexog",
        "SARIMAX_y1_exog",
        "SARIMAX_y2_noexog",
        "SARIMAX_y2_exog",
    ]
    all_names = ml_names + linear_names + sarimax_names

    errors = {h: {name: [] for name in all_names} for h in horizons}
    dates = {h: [] for h in horizons}

    # ROLLING
    for t_idx in tqdm(range(start_idx, len(X))):
        t = X.index[t_idx]

        for h in horizons:
            end_train_date = t - pd.DateOffset(months=h)
            if end_train_date <= X.index[0]:
                continue

            X_train = X.loc[:end_train_date]
            y_train = y.loc[:end_train_date]

            if fixed_train_window is not None:
                X_train = X_train.iloc[-fixed_train_window:]
                y_train = y_train.iloc[-fixed_train_window:]

            X_t = X.loc[[t]].values
            y_true = y.loc[[t]].values

            if verbose and (t_idx - start_idx) % 50 == 0 and h == 1:
                print(f"t={t.date()}, h={h}, train size={len(X_train)}")

            # XGBOOST RANDOMFOREST NEURALNET
            for idx, (name, model) in enumerate(
                [("RF", rf_model), ("XGB", xgb_model), ("NN", nn_model)]
            ):
                try:
                    model.fit(X_train.values, y_train.values)
                    y_pred = model.predict(X_t)
                    e = (y_pred - y_true)[0]
                except Exception as ex:
                    print(f"Erreur {name} à t={t.date()}, h={h} : {ex}")
                    e = np.array([np.nan, np.nan])
                errors[h][name].append(e)

            # LASSO RIDGE ELASTICNET DFM
            for name, make_fn in [
                ("Ridge", fit_ridge_ts_cv),
                ("Lasso", fit_lasso_ts_cv),
                ("ElasticNet", fit_elasticnet_ts_cv),
                ("DFM", fit_dfm_pca_ridge_ts_cv),
            ]:
                try:
                    model = make_fn(X_train.values, y_train.values)
                    y_pred = model.predict(X_t)  # (1, 2)
                    e = (y_pred - y_true)[0]  # (2,)
                except Exception as ex:
                    print(f"Erreur {name} à t={t.date()}, h={h} : {ex}")
                    e = np.array([np.nan, np.nan])
                errors[h][name].append(e)

            # SARIMAX
            X_future_exog = np.repeat(X_t, repeats=h, axis=0)

            for j in range(2):
                yj_train = y_train.iloc[:, j]

                for use_exog in [False, True]:
                    name = f"SARIMAX_y{j + 1}_{'exog' if use_exog else 'noexog'}"
                    X_train_exog = X_train.values if use_exog else None

                    try:
                        forecaster = SARIMAXForecaster(order=(2, 0, 1))
                        forecaster.fit(yj_train, X_train_exog=X_train_exog)
                        y_path = forecaster.predict_h(
                            X_future_exog=X_future_exog if use_exog else None, h=h
                        )
                        y_pred_t = y_path[-1]
                        e = y_pred_t - y_true[0, j]
                    except Exception as ex:
                        print(f"Erreur {name} à t={t.date()}, h={h} : {ex}")
                        e = np.nan
                    errors[h][name].append(e)

            dates[h].append(t)

    for h in horizons:
        for name in all_names:
            errors[h][name] = np.asarray(errors[h][name])

    # Comppute wins over the window
    wins = {h: {"rmse": {}, "mae": {}} for h in horizons}

    for h in horizons:
        n_eval = len(errors[h]["RF"])
        n_windows = n_eval - comparison_length + 1

        if n_windows <= 0:
            print(
                f"Horizon {h}: n_eval={n_eval}, comparison_length={comparison_length} : pas assez de fenêtres"
            )
            continue

        for name in all_names:
            wins[h]["rmse"][name] = 0
            wins[h]["mae"][name] = 0

        for w in range(n_windows):
            best_rmse_name = None
            best_mae_name = None
            best_rmse_val = np.inf
            best_mae_val = np.inf

            for name in all_names:
                e = errors[h][name][w : w + comparison_length]
                if isinstance(e, np.ndarray):
                    if e.ndim == 2:
                        e_valid = e[~np.isnan(e).any(axis=1)]
                        if len(e_valid) == 0:
                            continue
                        rmse = np.sqrt(np.mean(e_valid**2))
                        mae = np.mean(np.abs(e_valid))
                    else:
                        e_valid = e[~np.isnan(e)]
                        if len(e_valid) == 0:
                            continue
                        rmse = np.sqrt(np.mean(e_valid**2))
                        mae = np.mean(np.abs(e_valid))
                else:
                    continue

                if rmse < best_rmse_val:
                    best_rmse_val = rmse
                    best_rmse_name = name
                if mae < best_mae_val:
                    best_mae_val = mae
                    best_mae_name = name

            if best_rmse_name:
                wins[h]["rmse"][best_rmse_name] += 1
            if best_mae_name:
                wins[h]["mae"][best_mae_name] += 1

    return errors, wins, dates
