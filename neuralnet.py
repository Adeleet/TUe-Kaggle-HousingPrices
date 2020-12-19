# %%
import itertools

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib as mpl

# %%
# constant tensor
x = tf.constant([[5, 2], [1, 3]])
x
# %%
x.numpy()
# %%
# random constant tensor
y = tf.random.normal(shape=(3,3), mean=0.0)
y
# %%
# variable tensor with y as initial value
a = tf.Variable(y)
a
# %%
df_train_orig_csv = pd.read_csv("./data/train.csv.gz")
df_train_categorical = df_train_orig_csv.select_dtypes(include=['object'])
df_train_onehot = pd.get_dummies(df_train_categorical)
df_train_orig = df_train_orig_csv.select_dtypes(exclude=['object'])
df_train_orig.fillna(0, inplace=True)
df_train_orig.drop('Id', axis=1, inplace=True)
df_train = df_train_orig.join(df_train_onehot)
# %%
df_test_orig = pd.read_csv("./data/test.csv.gz")
df_test_categorical = df_test_orig.select_dtypes(include=['object'])
df_test_onehot = pd.get_dummies(df_test_categorical)
df_test = df_test_orig.select_dtypes(exclude=['object'])
df_test.fillna(0, inplace=True)
ID = df_test.Id
df_test.drop('Id', axis=1, inplace=True)
df_test = df_test.join(df_test_onehot)
# %%
df_train, df_test = df_train.align(df_test, axis=1)
df_test.drop('SalePrice', axis=1, inplace=True)
df_test.fillna(0, inplace=True)
df_train.fillna(0, inplace=True)
# %%
clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(df_train)
y_noano = clf.predict(df_train)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
y_noano[y_noano['Top'] == 1].index.values

df_train = df_train.iloc[y_noano[y_noano['Top'] == 1].index.values]
df_train.reset_index(drop = True, inplace = True)
print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
print("Number of rows without outliers:", df_train.shape[0])
# %%
FEATURES = list(df_train.columns)
FEATURES_TRAIN = list(df_train.columns)

FEATURES
FEATURES_TRAIN.remove('SalePrice')
# %%
mat_train = np.matrix(df_train)
# mat_test  = np.matrix(test)
mat_x = np.matrix(df_train.drop('SalePrice', axis = 1))
mat_y = np.matrix(df_train.SalePrice).reshape((-1,1))

prepro_y = MinMaxScaler()
prepro_y.fit(mat_y)

prepro = MinMaxScaler()
prepro.fit(mat_train)

prepro_test = MinMaxScaler()
prepro_test.fit(mat_x)
df_train = pd.DataFrame(prepro.transform(mat_train),columns = list(df_train.columns))
# %%
# List of features
COLUMNS = FEATURES

LABEL = "SalePrice"

# Columns
feature_cols = FEATURES_TRAIN

# Training set and Prediction set with the features to predict
training_set = df_train[COLUMNS]
prediction_set = df_train.SalePrice
# Train and Test 
x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES_TRAIN] , prediction_set, test_size=0.33, random_state=42)
y_train = pd.DataFrame(y_train, columns = [LABEL])
training_set = pd.DataFrame(x_train, columns = FEATURES_TRAIN).merge(y_train, left_index = True, right_index = True)
training_set.head()

# Training for submission
training_sub = training_set[COLUMNS]
# Same thing but for the test set
y_test = pd.DataFrame(y_test, columns = [LABEL])
testing_set = pd.DataFrame(x_test, columns = feature_cols).merge(y_test, left_index = True, right_index = True)
testing_set.head()
# %%
feature_cols = training_set[FEATURES_TRAIN]
labels = training_set["SalePrice"].values
# %%
model = tf.keras.Sequential()
model.add(Dense(300, input_dim=288, kernel_initializer='normal', activation='relu'))
model.add(Dense(300, kernel_initializer='normal', activation='relu'))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta())

model.fit(np.array(feature_cols).astype('float32'), np.array(labels), epochs=2000, batch_size=10)
# %%
model.evaluate(np.array(feature_cols), np.array(labels))
# %%
# Predictions
feature_cols_test = testing_set[FEATURES_TRAIN]
labels_test = testing_set[LABEL].values

y = model.predict(np.array(feature_cols_test))
predictions = list(itertools.islice(y, testing_set.shape[0]))

# predictions = prepro_y.inverse_transform(np.array(predictions).reshape(-1,1))
# reality = pd.DataFrame(prepro.inverse_transform(testing_set), columns = COLUMNS).SalePrice
reality = pd.DataFrame(testing_set, columns = COLUMNS).SalePrice
# %%
mpl.rc('xtick', labelsize=30) 
mpl.rc('ytick', labelsize=30) 

fig, ax = plt.subplots(figsize=(50, 40))

plt.style.use('ggplot')
plt.plot(predictions, reality, 'ro')
plt.xlabel('Predictions', fontsize = 30)
plt.ylabel('Reality', fontsize = 30)
plt.title('Predictions x Reality on dataset Test', fontsize = 30)
ax.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
# %%
mat_test = np.matrix(df_test)
test = pd.DataFrame(prepro_test.transform(mat_test),columns = FEATURES_TRAIN)
# %%
y_predict = model.predict(np.array(test))

def to_submit(pred_y,name_out):
    y_predict = list(itertools.islice(pred_y, test.shape[0]))
    y_predict = pd.DataFrame(prepro_y.inverse_transform(np.array(y_predict).reshape(len(y_predict),1)), columns = ['SalePrice'])
    y_predict = y_predict.join(ID)
    y_predict.to_csv(name_out + '.csv',index=False)
    
to_submit(y_predict, "submission_continuous")
# %%
