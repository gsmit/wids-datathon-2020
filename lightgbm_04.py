import time
import numpy as np
import pandas as pd
import sklearn as sk
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold

# disable lgb UserWarning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

PATH = 'C:/Users/GijsSmit/OneDrive - TU Eindhoven/kaggle/wids-datathon-2020'

version = '04'
date = time.strftime('%Y%m%d_%H%M')

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

# extract categorical features
cat_feats = df.select_dtypes(include=['object', 'category'])
cat_feats = cat_feats.columns.tolist()

cat_feats = cat_feats + ['ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source',
             'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem',
             'elective_surgery', 'apache_post_operative', 'arf_apache', 'gcs_eyes_apache',
             'gcs_motor_apache', 'gcs_unable_apache', 'gcs_verbal_apache', 'intubated_apache',
             'ventilated_apache', 'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure',
             'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis',
             'icu_id', 'apache_3j_diagnosis', 'apache_2_diagnosis']
cat_feats = list(set(cat_feats))

# remove features with only 1 or only unique values
df = df.drop(columns=['patient_id', 'readmission_status'], axis=1)

# dropping 'h1_bilirubin_max', 'h1_bilirubin_min', 'h1_albumin_min', 'h1_albumin_max'
# df = df.drop(['h1_bilirubin_max', 'h1_bilirubin_min', 'h1_albumin_min', 'h1_albumin_max'], axis=1)

for feat in df.columns:
    if '_min' in feat:
        base = feat.split('_min')[0]
        print('Adding avg for:', base)
        feat_min = feat
        feat_max = base + '_max'
        feat_avg = base + '_AVG'
        df[feat_avg] = df[feat_max] - df[feat_min]

# replace missing values indicated with -1
df["apache_4a_hospital_death_prob"] = df["apache_4a_hospital_death_prob"].replace({-1: np.nan})
df["apache_4a_icu_death_prob"] = df["apache_4a_icu_death_prob"].replace({-1: np.nan})

# frequency encoding  features
for feat in cat_feats:
    feat_count = feat + '_COUNT'
    null_count = df[feat].isnull().sum()
    df[feat_count] = df.groupby(feat)[feat].transform('count')
    df[feat_count] = df[feat_count].fillna(null_count)
print('Finished count encoding features')

# feature aggregations
feat_set_a = [
    'age',
    'apache_4a_hospital_death_prob',
    'apache_4a_icu_death_prob',
    'd1_lactate_min',
    'd1_spo2_min',
    'd1_heartrate_min',
    'd1_sysbp_min',
    'd1_platelets_min',
    'd1_lactate_max',
    'd1_sysbp_noninvasive_min',
    'd1_resprate_min',
    'd1_resprate_max',
    # 'd1_arterial_ph_min',
    # 'd1_bun_min',
    # 'd1_glucose_min'
]

feat_set_a = set(feat_set_a)  # make sure values are unique

feat_set_b = cat_feats + ['icu_id', 'hospital_id']

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

# label encoding
feat_num = 1
len_columns = len(df.columns)
for f in df.columns:
    if (df[f].dtype == 'object') or ((f in cat_feats) and (df[f].min() <= 0)):
        le = LabelEncoder()
        le.fit(list(df[f].fillna('missing').values))
        df[f] = le.transform(list(df[f].fillna('missing').values))
    if (feat_num % 100 == 0) or (feat_num == len_columns):
        print('Label encoded', feat_num, 'of', len_columns)
    feat_num = feat_num + 1
print('Finished label encoding')

# remove adversarial validation features that score high
df = df.drop(columns=['icu_id', 'hospital_id'])

df_copy = df.copy()
seed_scores = []
seed_predictions = []


for seed in [42, 96, 101, 128, 256]:

    print('Starting seed {}'.format(seed), '\n')
    df = df_copy.copy()

    # important parameters
    kfold_splits = 5
    total_num_round = 2200
    early_num_round = 200
    num_leaves = 2 ** 5
    learning_rate = 0.01

    params = {
        'boosting_type': 'gbdt',
        'boost_from_average': True,
        'objective': 'binary',
        'metric': {'auc'},
        'is_unbalance': False,
        'scale_pos_weight': 1.22,
        'max_depth': 17,
        'num_leaves': num_leaves,
        'min_data_in_leaf': 500,
        'learning_rate': learning_rate,
        # 'bagging_fraction': 0.75,
        # 'bagging_freq': 4,
        # 'bagging_seed': 42,
        'feature_fraction': 0.60,
        'feature_fraction_seed': seed,
        'lambda_l1': 4.17,
        'lambda_l2': 4.64,
        'seed': 42,
        'verbosity': -1}

    columns = list(df.columns)

    columns_to_drop = ['hospital_death', 'encounter_id']
    columns_to_drop = [value for value in columns_to_drop if value in df.columns]

    X = df.loc[df['is_train'] == 1].drop(columns_to_drop, axis=1).copy()
    y = df.loc[df['is_train'] == 1]['hospital_death'].copy()
    X_index = df.loc[df['is_train'] == 1].drop(columns_to_drop, axis=1).index.tolist()
    X_test = df.loc[df['is_train'] == 0].drop(columns_to_drop, axis=1).copy()
    del df

    # filter categorical features
    cat_feats = [value for value in list(X.columns) if value in cat_feats]

    sample_submission = pd.read_csv(PATH + '/data/solution_template.csv')

    kfold_prediction = np.zeros(len(sample_submission))
    kfold_importance = []
    kfold_auc_scores = []
    kfold_best_round = []

    feat_importance_split = []
    feat_importance_gain = []

    folds = KFold(n_splits=kfold_splits, shuffle=True, random_state=seed)
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

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)

        bst = lgb.train(
            params=params,
            train_set=lgb_train,
            num_boost_round=total_num_round,
            early_stopping_rounds=early_num_round,
            valid_sets=[lgb_valid],
            valid_names=['VALID'],
            categorical_feature=cat_feats,
            verbose_eval=200)

        feat_importance_split.append(bst.feature_importance(importance_type='split'))
        feat_importance_gain.append(bst.feature_importance(importance_type='gain'))

        prediction = bst.predict(X_test)
        kfold_prediction = kfold_prediction + (prediction / kfold_splits)

        importance = bst.feature_importance()
        kfold_importance.append(importance)

        auc_score = round(bst.best_score['VALID']['auc'], 5)
        kfold_auc_scores.append(auc_score)

        kfold_best_round.append(bst.best_iteration)

        lgb.plot_importance(bst, importance_type='split', max_num_features=75, figsize=(12, 12))
        plt.title('Feature importance (by SPLIT) fold = {} and seed = {}'.format(fold_num, seed))
        plt.show()

        lgb.plot_importance(bst, importance_type='gain', max_num_features=75, figsize=(12, 12))
        plt.title('Feature importance (by GAIN) fold = {} and seed = {}'.format(fold_num, seed))
        plt.show()

        del prediction, importance, lgb_train, lgb_valid
        fold_duration = time.time() - fold_start
        print('Fold {} finished in {} minutes'.format(fold_num, int(round(fold_duration / 60, 0))), '\n')

        # start with next fold
        fold_num += 1

    # make sample submission file
    auc_score_mean = round(float(np.mean(kfold_auc_scores)), 5)
    auc_score_std = round(float(np.std(kfold_auc_scores)), 5)
    print('Average AUC score on CV:  {} (STD: {})'.format(auc_score_mean, auc_score_std))

    num_feats = len(X_test.columns)
    kfold_prediction = np.clip(kfold_prediction, 0, 1)
    sample_submission['hospital_death'] = kfold_prediction

    sample_submission.to_csv(
        PATH + '/submissions/light_gbm_{}_mean_{}_std_{}_feats_{}_lr_{}_leaves_{}_seed_{}.csv'.format(version, date, auc_score_mean, auc_score_std, num_feats, learning_rate, num_leaves, seed), index=False)
    print('Finished saving mean of predictions')

    print('Finished with seed {}'.format(seed), '\n')

    # append to seed lists
    seed_scores.append(auc_score_mean)
    seed_predictions.append(kfold_prediction)


mean_seed_score = np.mean(seed_scores)
print('Average AUC score over all seeds is {}'.format(mean_seed_score))
sample_submission = pd.read_csv(PATH + '/data/solution_template.csv')

import numpy as np
mean_scores = (np.array(seed_predictions[0]) +
               np.array(seed_predictions[1]) +
               np.array(seed_predictions[2]) +
               np.array(seed_predictions[3]) +
               np.array(seed_predictions[4])) / 5.0
sample_submission['hospital_death'] = mean_scores
sample_submission.to_csv(
    PATH + '/submissions/light_gbm_{}_mean_{}_combined_score.csv'.format(version, date, mean_seed_score), index=False)
print('Finished saving mean of all seeds')
