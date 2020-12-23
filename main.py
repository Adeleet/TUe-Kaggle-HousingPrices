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
from preprocess import pre
import numpy as np
# pre-processing
from scipy import stats
from scipy.stats import norm, skew 
# section one hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer



# %%
DATA_BASE_URL = "https://github.com/Adeleet/TUe-Kaggle-HousingPrices/raw/main/data"
df_train = pd.read_csv(f"{DATA_BASE_URL}/train.csv.gz")
df_test = pd.read_csv(f"{DATA_BASE_URL}/test.csv.gz")

# %% 
test = df_test.copy()
train = df_train.copy()



# %% 

## Creating normal distribution of target variable to increase accuracy of linear models

# y = train['SalePrice']
# plt.figure(); plt.title('Log Normal')
# sns.distplot(y, kde=False, fit=stats.lognorm)


# train['SalePrice'] = np.log1p(train['SalePrice'])
# sns.distplot(train['SalePrice'], fit = norm)



#%%
quants = [col for col in train.columns if train.dtypes[col] != 'object']
quants.remove('SalePrice'), quants.remove('Id')
cats = [col for col in train.columns if train.dtypes[col] == 'object']

# %%
## PLOTTING NA VALUES
# CATS
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation = 90)
sns.barplot(cats, train[cats].isna().sum(), order=train[cats].isna().sum().sort_values().index[::-1]).set_title('Categorical variables - Na values')
# %%
# QUANTS
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation = 90)
sns.barplot(quants, train[quants].isna().sum(), order=train[quants].isna().sum().sort_values().index[::-1]).set_title('Quantitative variables - Na values')


###

# %%
# TODO -> REMOVING OUTLIERS FROM TRAINs
###
#from sklearn.ensemble import IsolationForest
# clf = IsolationForest(max_samples = 100, random_state = 42)
# clf.fit(df_train)
# y_noano = clf.predict(df_train)
# y_noano = pd.DataFrame(y_noano, columns = ['Top'])
# y_noano[y_noano['Top'] == 1].index.values

# df_train = df_train.iloc[y_noano[y_noano['Top'] == 1].index.values]
# df_train.reset_index(drop = True, inplace = True)
# print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
# print("Number of rows without outliers:", df_train.shape[0])


##
# TODO -> DIFFERENTIATE ORDINAL AND CATEGORICAL
# ???? 

cols = list(train.columns)
n = train.shape[0]
finalcols = []
for col in cols:
    if not any(train['PoolQC'].value_counts() > 0.99*n):
        finalcols += cols

# %%
# fixing categorical NA's
# train['PoolQC'].value_counts() > here we see the 4 categores, last one should be 'no pool'
train['PoolQC'].fillna('No pool', inplace = True)
test['PoolQC'].fillna('No pool', inplace = True)


## generic removal of NA's -> TODO improve with domain knowledge?
train[cats] = train[cats].apply(lambda x: x.fillna('None'))
test[cats] = test[cats].apply(lambda x: x.fillna('None'))


# %%
# fixing quant NA's by replacement -> TODO perhaps imputation by regression is better

# TODO Non generic solution
train[quants] = train[quants].apply(lambda x: x.fillna(0))
test[quants] = test[quants].apply(lambda x: x.fillna(0))



# %% 
# TODO -> SKEW FEATURES TO IMPROVE LINEAR MODELS
### SKLEARN - powertransform
pt = PowerTransformer() 
pt.fit(train[quants])
train[quants] = pt.transform(train[quants])



# %% 
# ONE HOT ENCODING VARIABLES
# -> this makes sure no 'objects' remain in the train/test dataframes


### TODO FIX THIS

encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(train[cats])


X = train.drop("SalePrice", axis=1)
y = train["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_en = encoder.transform(X_train[cats])
X_test_en = encoder.transform(X_test[cats])



# transformed = jobs_encoder.transform(data['Profession'].to_numpy().reshape(-1, 1))
# #Create a Pandas DataFrame of the hot encoded column
# ohe_df = pd.DataFrame(transformed, columns=jobs_encoder.get_feature_names())
# #concat with original data
# data = pd.concat([data, ohe_df], axis=1).drop(['Profession'], axis=1)






####
# old stuff



# %%
CORR_THRESHOLD = 0.4  # vars should have at least this correlation to be considered as predictor
var_corr = (
    df_train.corr()["SalePrice"].abs().sort_values(ascending=False)
)  # get correlation of each var with dependent var (SalePrice)
predictor_names = var_corr[
    (var_corr > CORR_THRESHOLD) & (var_corr < 1)
].index  # get predictor names above threshold


# %%
train_data = df_train[list(predictor_names) + ["SalePrice"]]
test_data = df_test[list(predictor_names)]


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
