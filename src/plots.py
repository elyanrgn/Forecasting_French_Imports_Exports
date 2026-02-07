import warnings
from model_saving import load_model
from rolling_eval import rolling_window_complete
import pandas as pd

warnings.filterwarnings("ignore")

model_rf = load_model("random_forest_model")
model_xgb = load_model("xgboost_model")
model_nn = load_model("neural_network_model")


data = pd.read_csv(
    "data\\processed\\FRdataM_features.csv", index_col=0, parse_dates=True
)

y = data[["Imports", "Exports"]]
X = data.drop(columns=["Imports", "Exports"])


errors, wins, dates = rolling_window_complete(
    X,
    y,
    model_rf,
    model_xgb,
    model_nn,
    horizons=(1, 3, 6, 12),
    start_test=None,
    comparison_length=24,  # 2 years (monthly)
    fixed_train_window=None,
    verbose=True,
)

print("ROLLING WINDOW RESULTS(wins over 24 months)")
for h in (1, 3, 6, 12):
    print(f"Horizon h={h} months")
    print("Metric: RMSE")
    total_wins_rmse = sum(wins[h]["rmse"].values())
    for name in sorted(wins[h]["rmse"].keys()):
        w = wins[h]["rmse"][name]
        pct = 100 * w / total_wins_rmse if total_wins_rmse > 0 else 0
        print(f"  {name} : {w:3d} wins ({pct:5.1f}%)")
    print("Metric: MAE")
    total_wins_mae = sum(wins[h]["mae"].values())
    for name in sorted(wins[h]["mae"].keys()):
        w = wins[h]["mae"][name]
        pct = 100 * w / total_wins_mae if total_wins_mae > 0 else 0
        print(f"  {name} : {w:3d} wins ({pct:5.1f}%)")
