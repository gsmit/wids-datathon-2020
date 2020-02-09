import time
import numpy as np
import pandas as pd
import sklearn as sk
import lightgbm as lgb

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold

PATH = 'C:/Users/GijsSmit/OneDrive - TU Eindhoven/kaggle/wids-datathon-2020'

version = '01'
date = time.strftime('%Y%m%d_%H%M')


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


# load train and test set
train = pd.read_csv('data/training_v2.csv')
test = pd.read_csv('data/unlabeled.csv')

# add train indicator
train['is_train'] = 1
test['is_train'] = 0

# concatenate train and test for feature engineering
df = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
print('Shape of train:', train.shape)
print('Shape of test:', test.shape)
print('Shape of df:', df.shape)
del train, test

print()
print(df.shape)

# group similar categories
df['hospital_admit_source'] = df['hospital_admit_source'].replace(
    {'Other ICU': 'ICU', 'ICU to SDU': 'SDU', 'Step-Down Unit (SDU)': 'SDU', 'Other Hospital': 'Other',
     'Observation': 'Recovery Room', 'Acute Care/Floor': 'Acute Care'})
df['icu_type'] = df['icu_type'].replace({'CCU-CTICU': 'Grpd_CICU', 'CTICU': 'Grpd_CICU', 'Cardiac ICU': 'Grpd_CICU'})
df['apache_2_bodysystem'] = df['apache_2_bodysystem'].replace({'Undefined diagnoses': 'Undefined Diagnoses'})

# replace missing values indicated with -1
df["apache_4a_hospital_death_prob"] = df["apache_4a_hospital_death_prob"].replace({-1: np.nan})
df["apache_4a_icu_death_prob"] = df["apache_4a_icu_death_prob"].replace({-1: np.nan})

# extract categorical features
cat_feats = df.select_dtypes(include=['object', 'category'])
cat_feats = cat_feats.columns.tolist()

columns = list(df.columns)
columns_to_drop = ['hospital_death', 'hospital_id', 'encounter_id', 'readmission_status']
columns_to_drop = [value for value in columns_to_drop if value in df.columns]
cat_feats = [value for value in cat_feats if value not in columns_to_drop]

for feat in cat_feats:
    dummies = pd.get_dummies(df[feat], drop_first=False, dummy_na=True, prefix='onehot_' + feat)
    dummies.columns = [s.lower().replace(' ', '_') for s in dummies.columns]
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[feat])

drop_count = 1
for feat in df.columns:
    pct_missing = df[feat].isna().sum() / len(df[feat])
    if (pct_missing >= 0.25) and (feat != 'hospital_death'):
        df = df.drop(columns=[feat])
        print('Dropped column {} ({}% missing)'.format(feat, round(pct_missing * 100, 3)))
        drop_count += 1
print('Dropped {} columns in total'.format(drop_count))

for feat in df.columns:
    df[feat] = df[feat].fillna(df[feat].median())

print()
print(df.shape)


X = df.loc[df['is_train'] == 1].drop(columns_to_drop, axis=1).copy().reset_index(drop=True)
y = df.loc[df['is_train'] == 1]['hospital_death'].copy().reset_index(drop=True)
X_index = df.loc[df['is_train'] == 1].drop(columns_to_drop, axis=1).index.tolist()
X_test = df.loc[df['is_train'] == 0].drop(columns_to_drop, axis=1).copy().reset_index(drop=True)
del df

# filter categorical features
cat_feats = [value for value in list(X.columns) if value in cat_feats]

sample_submission = pd.read_csv(PATH + '/data/solution_template.csv')

kfold_prediction = np.zeros(len(sample_submission))
kfold_auc_scores = []

feat_importance_split = []
feat_importance_gain = []

kfold_splits = 5
folds = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)
splits = folds.split(X, y)

# create kfold splits
fold_num = 1
for train_index, valid_index in splits:
    fold_start = time.time()
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    print('fold number:  ', fold_num)
    print('train_index:  ', train_index[0], 'to', train_index[-1])
    print('valid_index:  ', valid_index[0], 'to', valid_index[-1])
    print('X_train.shape:', X_train.shape)
    print('X_valid.shape:', X_valid.shape)
    print('# of features:', len(X_train.columns))

    input_dim = X_train.shape[1]

    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auroc])

    model.fit(X_train, y_train, epochs=5, batch_size=128)
    score = model.evaluate(X_valid, y_valid, batch_size=128)[1]
    print('Score:', score)

    y_pred = model.predict(X_valid)
    auc_score = roc_auc_score(y_valid, y_pred)

    prediction = model.predict(X_test)
    kfold_prediction = kfold_prediction + (prediction / kfold_splits)

    kfold_auc_scores.append(auc_score)

    del prediction
    fold_duration = time.time() - fold_start
    print('Fold {} AUC: {}'.format(fold_num, round(auc_score, 4)))
    print('Fold {} finished in {} minutes'.format(fold_num, int(round(fold_duration / 60, 0))), '\n')

    # start with next fold
    fold_num += 1

# make sample submission file
auc_score_mean = round(float(np.mean(kfold_auc_scores)), 4)
auc_score_std = round(float(np.std(kfold_auc_scores)), 4)
print('Average AUC score on CV:  {} (STD: {})'.format(auc_score_mean, auc_score_std))

num_feats = len(X_test.columns)
kfold_prediction = np.clip(kfold_prediction, 0, 1)
sample_submission['hospital_death'] = kfold_prediction

#%%
sample_submission.to_csv(
    PATH + '/submissions/keras_mlp_{}_mean_{}_std_{}_feats_{}.csv'.format(version, date, auc_score_mean, auc_score_std, num_feats), index=False)
print('Finished saving mean of predictions')

#%%
auc_score = roc_auc_score(y_valid.values, y_pred)


