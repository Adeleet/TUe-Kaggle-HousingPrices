# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
# # 1 Preprocessing

# %%
# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# %%
DATA_BASE_URL = "https://github.com/Adeleet/TUe-Kaggle-HousingPrices/raw/main/data"
# df_train = pd.read_csv(f"{DATA_BASE_URL}/train.csv.gz")
# df_test = pd.read_csv(f"{DATA_BASE_URL}/test.csv.gz")
df_train = pd.read_csv("./data/train.csv.gz")
df_test = pd.read_csv("./data/test.csv.gz")


# %% [markdown]
#### Convert numerical variables to categorical

# %%
df_train.plot.scatter(x="OverallQual",y="SalePrice")
df_train["OverallQual"] = df_train["OverallQual"].astype("str")
# %%
CORR_THRESHOLD = 0.4  # vars should have at least this correlation to be considered as predictor
var_corr = (
    df_train.corr()["SalePrice"].abs().sort_values(ascending=False)
)  # get correlation of each var with dependent var (SalePrice)
predictor_names = var_corr[
    (var_corr > CORR_THRESHOLD) & (var_corr < 1)
].index  # get predictor names above threshold

predictor_names = list(predictor_names) + ["OverallQual"]
# %%
train_data = df_train[predictor_names + ["SalePrice"]]
test_data = df_test[predictor_names]

# %%
pd.get_dummies(train_data)

# %%
for dataset in [train_data, test_data]:
    dataset["hasGarage"] = (dataset["GrLivArea"] > 0).astype(
        "int"
    )  # create boolean variables indicating if garage/masonry exists
    dataset["hasMasonry"] = (dataset["MasVnrArea"] > 0).astype("int")
    dataset.fillna(0, inplace=True)  # fill NaN (garageArea is NaN when there is no garage)


# %%
X = train_data.drop("SalePrice", axis=1)
y = train_data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_true = pd.DataFrame(y_test)
y_true["Prediction"] = reg.predict(X_test)
y_true = np.log(y_true)
SSE = (y_true["SalePrice"] - y_true["Prediction"]) ** 2
RMSE = (SSE.sum() / SSE.shape[0]) ** 0.5


# %%
train_data.plot.scatter(x="GrLivArea", y="SalePrice")


# %%
df_submission = df_test[["Id"]]
df_submission["SalePrice"] = pred = reg.predict(test_data)


# %%
df_submission.to_csv("submissions/submission_test.csv.gz", index=False)

# %%

