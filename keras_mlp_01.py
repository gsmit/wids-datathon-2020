import time
import random
import numpy as np
import pandas as pd
import sklearn as sk
import lightgbm as lgb

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold

PATH = 'C:/Users/GijsSmit/OneDrive - TU Eindhoven/kaggle/wids-datathon-2020'

version = '01'
date = time.strftime('%Y%m%d_%H%M')


def auroc(y_true, y_prediction):
    return tf.py_func(roc_auc_score, (y_true, y_prediction), tf.double)


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

# feature aggregations
feat_set_a = ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']
feat_set_a = set(feat_set_a)  # make sure values are unique
feat_set_b = cat_feats

count = 1
feat_set_len = len(feat_set_a) * len(feat_set_b)
for feat_a in feat_set_a:
    for feat_b in feat_set_b:
        feat_name_mean = feat_a + '_' + feat_b + '_AGG_MEAN'
        feat_name_diff = feat_a + '_' + feat_b + '_AGG_DIFF'
        feat_name_std = feat_a + '_' + feat_b + '_AGG_STD'

        df[feat_name_mean] = df[feat_a] / df[[feat_a, feat_b]].copy().fillna(-999).groupby([feat_b])[feat_a].transform('mean')
        df[feat_name_diff] = df[feat_a] - df[[feat_a, feat_b]].copy().fillna(-999).groupby([feat_b])[feat_a].transform('mean')
        df[feat_name_std] = df[feat_a] / df[[feat_a, feat_b]].copy().fillna(-999).groupby([feat_b])[feat_a].transform('std')

        if (count % 1 == 0) or (count == feat_set_len):
            print('Finished aggregating feature', count, 'of', feat_set_len, '({} + {})'.format(feat_a, feat_b))

        count += 1

print('Finished creating aggregated features', '\n')
print('Number of total features:', len(df.columns))

columns = list(df.columns)
columns_to_drop = ['hospital_death', 'hospital_id', 'icu_id', 'encounter_id', 'readmission_status']
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
    if (pct_missing >= 0.75) and (feat != 'hospital_death'):
        df = df.drop(columns=[feat])
        print('Dropped column {} ({}% missing)'.format(feat, round(pct_missing * 100, 3)))
        drop_count += 1
print('Dropped {} columns in total'.format(drop_count))

for feat in df.columns:
    df[feat] = df[feat].fillna(df[feat].median())

print()
print(df.shape)

# standardize features
for feat in df.columns:
    scaler = MinMaxScaler()
    scaler.fit(df[[feat]].values)
    df[feat] = scaler.transform(df[[feat]].values)
print('Finished standardizing features')

X = df.loc[df['is_train'] == 1].drop(columns_to_drop, axis=1).copy().reset_index(drop=True)
y = df.loc[df['is_train'] == 1]['hospital_death'].copy().reset_index(drop=True)
X_index = df.loc[df['is_train'] == 1].drop(columns_to_drop, axis=1).index.tolist()
X_test = df.loc[df['is_train'] == 0].drop(columns_to_drop, axis=1).copy().reset_index(drop=True)
X_0 = df.loc[(df['is_train'] == 1) & (df['hospital_death'] == 0)].drop(columns_to_drop, axis=1).copy().reset_index(drop=True)
y_0 = df.loc[(df['is_train'] == 1) & (df['hospital_death'] == 0)]['hospital_death'].copy().reset_index(drop=True)
X_1 = df.loc[(df['is_train'] == 1) & (df['hospital_death'] == 1)].drop(columns_to_drop, axis=1).copy().reset_index(drop=True)
y_1 = df.loc[(df['is_train'] == 1) & (df['hospital_death'] == 1)]['hospital_death'].copy().reset_index(drop=True)
del df

# filter categorical features
cat_feats = [value for value in list(X.columns) if value in cat_feats]

sample_submission = pd.read_csv(PATH + '/data/solution_template.csv')

kfold_prediction = np.zeros(len(sample_submission))
kfold_auc_scores = []

kfold_splits = 5
# folds = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)
# splits = folds.split(X_1, y_1)

folds = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)
splits = folds.split(X, y)

FINAL_PREDICT = False

# create kfold splits
fold_num = 1
for train_index, valid_index in splits:
    fold_start = time.time()

    # random.seed(a=fold_num)
    # sample_factor = len(X_1) * 2
    # sample_indices = random.sample([i for i in range(len(X_0))], sample_factor)
    # X_0_sample = X_0.iloc[sample_indices].copy().reset_index(drop=True)
    # y_0_sample = y_0.iloc[sample_indices].copy().reset_index(drop=True)
    #
    # X_0_train = X_0_sample.iloc[train_index]
    # X_0_valid = X_0_sample.iloc[valid_index]
    # X_1_train = X_1.iloc[train_index]
    # X_1_valid = X_1.iloc[valid_index]
    #
    # y_0_train = y_0_sample.iloc[train_index]
    # y_0_valid = y_0_sample.iloc[valid_index]
    # y_1_train = y_1.iloc[train_index]
    # y_1_valid = y_1.iloc[valid_index]
    #
    # X_train = pd.concat([X_0_train, X_1_train], ignore_index=True)
    # X_valid = pd.concat([X_0_valid, X_1_valid], ignore_index=True)
    # y_train = pd.concat([y_0_train, y_1_train], ignore_index=True)
    # y_valid = pd.concat([y_0_valid, y_1_valid], ignore_index=True)
    # del X_0_train, X_0_valid, X_1_train, X_1_valid
    # del y_0_train, y_0_valid, y_1_train, y_1_valid

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
    model.add(Dropout(0.45))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=1.0)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=[auroc])

    model.fit(X_train, y_train, epochs=125, batch_size=4096, verbose=2)
    score = model.evaluate(X_valid, y_valid, batch_size=4096)[1]
    print('Score:', score)

    y_pred = model.predict(X_valid)
    auc_score = roc_auc_score(y_valid, y_pred)
    kfold_auc_scores.append(auc_score)

    if FINAL_PREDICT:
        prediction = model.predict(X_test)
        kfold_prediction = kfold_prediction + (prediction / kfold_splits)
        del prediction

    fold_duration = time.time() - fold_start
    print('Fold {} AUC: {}'.format(fold_num, round(auc_score, 5)))
    print('Fold {} finished in {} minutes'.format(fold_num, int(round(fold_duration / 60, 0))), '\n')

    # start with next fold
    fold_num += 1

# make sample submission file
auc_score_mean = round(float(np.mean(kfold_auc_scores)), 5)
auc_score_std = round(float(np.std(kfold_auc_scores)), 5)
print('Average AUC score on CV:  {} (STD: {})'.format(auc_score_mean, auc_score_std))

#%%
num_feats = len(X_test.columns)
kfold_prediction = np.clip(kfold_prediction, 0, 1)
sample_submission['hospital_death'] = kfold_prediction

sample_submission.to_csv(
    PATH + '/submissions/keras_mlp_{}_mean_{}_std_{}_feats_{}.csv'.format(version, date, auc_score_mean, auc_score_std, num_feats), index=False)
print('Finished saving mean of predictions')
