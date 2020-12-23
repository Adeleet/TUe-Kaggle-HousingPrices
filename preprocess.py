import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer

def pre(train, test):
    quants = [col for col in train.columns if train.dtypes[col] != 'object']
    quants.remove('SalePrice'), quants.remove('Id')
    cats = [col for col in train.columns if train.dtypes[col] == 'object']

    ##
    # TODO -> DIFFERENTIATE ORDINAL AND CATEGORICAL
    # ???? 

    # replacing missing values
    # TODO Non generic solution
    train['PoolQC'].fillna('No pool', inplace = True)
    test['PoolQC'].fillna('No pool', inplace = True)
    train[cats] = train[cats].apply(lambda x: x.fillna('None'))
    test[cats] = test[cats].apply(lambda x: x.fillna('None'))
    train[quants] = train[quants].apply(lambda x: x.fillna(0))
    test[quants] = test[quants].apply(lambda x: x.fillna(0))

    ### Removing outliers
    clf = IsolationForest(max_samples = 100, random_state = 42)
    clf.fit(train[quants])
    y_noano = clf.predict(train[quants])
    y_noano = pd.DataFrame(y_noano, columns = ['Top'])
    y_noano[y_noano['Top'] == 1].index.values

    train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
    train.reset_index(drop = True, inplace = True)
    print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
    print("Number of rows without outliers:", train.shape[0])


    #Transform variables for better linear models
    pt = PowerTransformer() 
    pt.fit(train[quants])
    train[quants] = pt.transform(train[quants])
    test[quants] = pt.transform(test[quants])



    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(train[cats].to_numpy().reshape(-1, 1))
    
    transtrain = encoder.transform(train[cats].to_numpy().reshape(-1, 1))


    X = train.drop("SalePrice", axis=1)
    y = train["SalePrice"]
    #encode 

    # transformed = jobs_encoder.transform(data['Profession'].to_numpy().reshape(-1, 1))
    # #Create a Pandas DataFrame of the hot encoded column
    # ohe_df = pd.DataFrame(transformed, columns=jobs_encoder.get_feature_names())
    # #concat with original data
    # data = pd.concat([data, ohe_df], axis=1).drop(['Profession'], axis=1)
    return test, train


## get skewed cols to remove them later
def getSkewedCols(data, dropoff=0.99):
    ## Remove uberskewed
    cols = list(data.columns)
    n = data.shape[0]
    finalcols = []
    for col in cols:
        if not any(data[col].value_counts() > dropoff*n):
            finalcols.append(cols)
    return finalcols