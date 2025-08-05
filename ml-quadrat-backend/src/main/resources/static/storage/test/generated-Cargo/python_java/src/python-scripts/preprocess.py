prediction_columns = ["prediction"]
import sys
import os
import pickle
import time, datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer

dataset = sys.argv[1]
sequential = sys.argv[2]
timestamps = sys.argv[3]
features = sys.argv[4].split(',')
feature_types = sys.argv[5].split(',')
labels = sys.argv[6]

from collections import Counter

col_names = []
num_col_names = []
cat_col_names = []

if timestamps.lower() == 'on':
    col_names.append('timestamp')

for i in range(len(features)):
    feature = features[i]
    feature_type = feature_types[i]
    if 'String' in feature_type or 'Char' in feature_type:
        cat_col_names.append(feature)
    if 'Int' in feature_type or 'Double' in feature_type:
        num_col_names.append(feature)
    col_names.append(feature)

raw_df = pd.read_csv(dataset, header=None, dtype=str, skip_blank_lines=True, on_bad_lines='skip')
raw_df.columns = col_names + [f'extra_col_{i}' for i in range(len(raw_df.columns) - len(col_names))]
df = raw_df.drop(columns=[col for col in raw_df.columns if col in prediction_columns], errors='ignore')
df = df[[col for col in df.columns if col in col_names]]

original_df = df.copy(deep=True)
with open('/home/uwubuntu/docker-full-stack/ml-quadrat-backend/src/main/resources/static/storage/test/generated-Cargo/python_java/src/python-scripts/pickles/preprocess_original_df.pickle', 'wb') as pickle_file:
    pickle.dump(original_df, pickle_file)

if(timestamps.lower() == 'on'):
    timeformat = "%d-%m-%Y %H:%M:%S"
    with open('/home/uwubuntu/docker-full-stack/ml-quadrat-backend/src/main/resources/static/storage/test/generated-Cargo/python_java/src/python-scripts/pickles/preprocess_timeformat.pickle', 'wb') as pickle_file:
        pickle.dump(timeformat, pickle_file)
    df.timestamp = df.timestamp.apply(lambda x: datetime.datetime.strptime(x, timeformat))
    df.set_index('timestamp', inplace=True)

for col in num_col_names:
    df[col] = pd.to_numeric(df[col], errors='coerce')

plots_path = '/home/uwubuntu/docker-full-stack/ml-quadrat-backend/src/main/resources/static/storage/test/generated-Cargo/python_java/src/python-scripts/plots/'

if len(cat_col_names) != 0:
    cat_col_names = [col for col in cat_col_names if col != 'timestamp']
    le = LabelEncoder()
    for col in cat_col_names:
        df[col] = le.fit_transform(df[col])
    with open('/home/uwubuntu/docker-full-stack/ml-quadrat-backend/src/main/resources/static/storage/test/generated-Cargo/python_java/src/python-scripts/pickles/preprocess_label_encoder.pickle', 'wb') as pickle_file:
        pickle.dump(le, pickle_file)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df[num_col_names])
df[num_col_names] = scaler.transform(df[num_col_names])

# Context-Specific Adjustments

# ===== non-timeseries unsupervised learning logic (labels ON) =====
if str(timestamps).strip().lower() == 'on':
    X_train = df.loc[:, col_names[1:-1]]
else:
    X_train = df.loc[:, col_names[:-1]]
y_train = df[features[-1]]

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of y_train: {y_train.shape}')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
print('X_train, X_test, y_train, y_test created in unsupervised mode')
print(f'Shape of X_train after split: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train after split: {y_train.shape}')
print(f'Shape of y_test: {y_test.shape}')

with open('/home/uwubuntu/docker-full-stack/ml-quadrat-backend/src/main/resources/static/storage/test/generated-Cargo/python_java/src/python-scripts/pickles/preprocess_X_train.pickle', 'wb') as pickle_file:
    pickle.dump(X_train, pickle_file)

with open('/home/uwubuntu/docker-full-stack/ml-quadrat-backend/src/main/resources/static/storage/test/generated-Cargo/python_java/src/python-scripts/pickles/preprocess_X_test.pickle', 'wb') as pickle_file:
    pickle.dump(X_test, pickle_file)

with open('/home/uwubuntu/docker-full-stack/ml-quadrat-backend/src/main/resources/static/storage/test/generated-Cargo/python_java/src/python-scripts/pickles/preprocess_y_train.pickle', 'wb') as pickle_file:
    pickle.dump(y_train, pickle_file)

with open('/home/uwubuntu/docker-full-stack/ml-quadrat-backend/src/main/resources/static/storage/test/generated-Cargo/python_java/src/python-scripts/pickles/preprocess_y_test.pickle', 'wb') as pickle_file:
    pickle.dump(y_test, pickle_file)

