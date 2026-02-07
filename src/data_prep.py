import pandas as pd
import numpy as np
from tempdisagg import TempDisaggModel
from sklearn.decomposition import PCA


def from_quarterly_to_monthly(dataset, info, method="ensemble"):
    quaterly_colomns = info.loc[dataset.columns]["Frequency"] == "Q"
    monthly_colomns = info.loc[dataset.columns]["Frequency"] == "M"
    stock_flux_info = info.loc[dataset.columns]["Type"]
    conversion_type = {"S": "average", "F": "average"}

    # Find the best indicator from the monthly columns for each quarterly column : in terms of correlation after resampling the monthly to quarterly
    mapping = {}
    for q_col in dataset.columns[quaterly_colomns]:
        best_corr = 0
        best_m_col = None
        for m_col in dataset.columns[monthly_colomns]:
            if stock_flux_info[q_col] == stock_flux_info[m_col]:
                corr = (dataset[q_col].resample("QE").mean()).corr(
                    dataset[m_col].resample("QE").mean()
                )
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_m_col = m_col
        mapping[q_col] = best_m_col
    # Convert quarterly columns to monthly using the best monthly indicator found
    result = {}
    index = (dataset.index).to_series().apply(lambda x: x.year).values
    grain = [i % 12 + 1 for i in range(len(index))]
    for q_col, m_col in mapping.items():
        series_type = stock_flux_info[q_col]
        if m_col is None:
            model = TempDisaggModel(
                method=method, conversion=conversion_type[series_type]
            )
            df = pd.DataFrame({"Index": index, "Grain": grain, "y": dataset[q_col]})
            model.fit(df)
            y_m = model.predict()
        else:
            model = TempDisaggModel(
                method=method, conversion=conversion_type[series_type]
            )
            df = pd.DataFrame(
                {
                    "Index": index,
                    "Grain": grain,
                    "y": dataset[q_col],
                    "X": dataset[m_col],
                }
            )
            model.fit(df)
            y_m = model.predict()
        result[q_col] = y_m
    m_index = dataset.index
    results_flat = {k: np.asarray(v).reshape(-1) for k, v in result.items()}
    df_q_to_m = pd.DataFrame.from_dict(results_flat, orient="columns")
    df_q_to_m = df_q_to_m[: len(m_index)]
    df_q_to_m.index = m_index[: len(df_q_to_m)]  # to be sure of same format
    df_final = pd.concat([df_q_to_m, dataset.loc[:, monthly_colomns]], axis=1)
    return df_final


data_FR = pd.read_excel("data\\raw\\FRdata.xlsx", sheet_name="data", index_col=0)
info = pd.read_excel("data\\raw\\FRdata.xlsx", sheet_name="info", index_col=0)


df_monthly = from_quarterly_to_monthly(data_FR, info)
# ZLB for index with negative values
df_monthly.dropna(how="all", inplace=True)
df_monthly = df_monthly.where(df_monthly >= 0, 0)
info_monthly = info.copy()
info_monthly.loc[info_monthly["Frequency"] == "Q", "Frequency"] = "M"
info_monthly.drop(columns=["Type"], inplace=True)

with pd.ExcelWriter(
    "data\\processed\\FRdata_monthly.xlsx", engine="xlsxwriter"
) as writer:
    df_monthly.to_excel(writer, sheet_name="data", index=True)
    info_monthly.to_excel(writer, sheet_name="info", index=True)
    # This file "FRdata_monthly.xlsx" contains the monthly dataset obtained after temporal disaggregation : it has to be fed to the Matlab code for further processing
    # The Matlab code will output the processed dataset in "FRdataM_COV_HT.xlsx"


## Adjusting data : include 4 lags for each variable and 4 first pcas
data = pd.read_excel(
    "data\\processed\\FRdataM_COV_HT.xlsx", index_col=0, parse_dates=True
)
pca = PCA(n_components=4)
pca_components = pca.fit_transform(data.drop(columns=["EXPGS_FR", "IMPGS_FR"]).dropna())
print("Explained variance ratio by the 4 first components :", pca_components.shape)
for i in range(4):
    data[f"PCA_{i + 1}"] = np.nan
    data.loc[data.drop(columns=["EXPGS_FR", "IMPGS_FR"]).index, f"PCA_{i + 1}"] = (
        pca_components[:, i]
    )

for col in data.columns:
    if col not in ["PCA_1", "PCA_2", "PCA_3", "PCA_4"]:
        for lag in range(1, 5):
            data[f"{col}_lag{lag}"] = data[col].shift(lag)


data.to_csv("data\\processed\\FRdataM_features.csv")
