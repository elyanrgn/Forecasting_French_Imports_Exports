from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class NeuralNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def make_optimizer(name, params, lr, weight_decay=0.0):
    if name == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "RMSprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "SGD":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(name)


def make_criterion(name, trial=None):
    if name == "MSE":
        return nn.MSELoss()
    if name == "MAE":
        return nn.L1Loss()
    if name == "Huber":
        delta = trial.suggest_float("huber_delta", 0.5, 5.0)
        return nn.HuberLoss(delta=delta)
    raise ValueError(name)


def rmse_torch(yhat, y):
    return torch.sqrt(nn.MSELoss()(yhat, y))


def objective(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    num_layers = trial.suggest_int("num_layers", 1, 10)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    loss_name = trial.suggest_categorical("loss", ["MSE", "MAE", "Huber"])

    layers = [X_train_tensor.shape[1]] + [hidden_size] * num_layers + [2]
    model = NeuralNet(layers).to(X_train_tensor.device)

    train_criterion = make_criterion(loss_name, trial)
    optimizer = make_optimizer(
        optimizer_name, model.parameters(), lr=lr, weight_decay=weight_decay
    )

    n_epochs = trial.suggest_int("epochs", 50, 300)

    train_hist, val_hist, val_rmse_hist = [], [], []
    best_val_rmse = float("inf")

    for epoch in range(n_epochs):
        # train
        model.train()
        optimizer.zero_grad()
        yhat = model(X_train_tensor)
        loss = train_criterion(yhat, y_train_tensor)
        loss.backward()
        optimizer.step()
        train_hist.append(loss.item())

        # val (loss + rmse)
        model.eval()
        with torch.no_grad():
            vhat = model(X_val_tensor)
            vloss = train_criterion(vhat, y_val_tensor).item()
            vrmse = rmse_torch(vhat, y_val_tensor).item()

        val_hist.append(vloss)
        val_rmse_hist.append(vrmse)
        best_val_rmse = min(best_val_rmse, vrmse)

        trial.report(vrmse, step=epoch)

    trial.set_user_attr("train_losses", train_hist)
    trial.set_user_attr("val_losses", val_hist)
    trial.set_user_attr("val_rmse", val_rmse_hist)

    return best_val_rmse


def objective_rf(trial, X_train, y_train, X_val, y_val):
    """X_train and y_train are defined in the global scope for simplicity, but you can pass them as arguments if needed."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 8, 12, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 0.8]
        ),
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
    return val_rmse


def objective_xgboost(trial, X_train, y_train, X_val, y_val):
    params = {
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "n_estimators": 5000,
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-2, 50.0, log=True
        ),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
    }

    model = XGBRegressor(**params, early_stopping_rounds=50)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    y_val_pred = model.predict(X_val)
    val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5

    trial.set_user_attr("best_iteration", int(model.best_iteration))

    return val_rmse
