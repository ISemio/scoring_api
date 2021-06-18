# Based on excellent script by @olivier
# https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm
#
# Additions and changes by Tilii:
# StratifiedKFold instead of KFold
# LightGBM parameters found by Bayesian optimization ( https://github.com/fmfn/BayesianOptimization )
#
# My additions and changes:
# Combining random oversampling and undersampling for imbalanced data
# Additionnal evaluation metrics: f1_score, precision, confusion matrix
# New function for prediction


import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import gc
import re
import pickle
from datetime import datetime
from collections import Counter

path = r"C:\Users\IS\Documents\Data Scientist\P7\P7_semionov_irina\P7_02_dossier\api\data\input\\"

def build_model_input():

    # bureau_balance
    buro_bal = pd.read_csv(path +'bureau_balance.csv')
    print('Buro bal shape : ', buro_bal.shape)

    print('transform to dummies')
    buro_bal = pd.concat(
        [buro_bal, pd.get_dummies(buro_bal.STATUS, prefix='buro_bal_status')],
        axis=1).drop(
            'STATUS', axis=1)

    print('Counting buros')
    buro_counts = buro_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
    buro_bal['buro_count'] = buro_bal['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])

    print('averaging buro bal')
    avg_buro_bal = buro_bal.groupby('SK_ID_BUREAU').mean()

    avg_buro_bal.columns = ['avg_buro_' + f_ for f_ in avg_buro_bal.columns]
    del buro_bal
    gc.collect()

    # bureau
    print('Read Bureau')
    buro = pd.read_csv(path + 'bureau.csv')

    print('Go to dummies')
    buro_credit_active_dum = pd.get_dummies(buro.CREDIT_ACTIVE, prefix='ca_')
    buro_credit_currency_dum = pd.get_dummies(buro.CREDIT_CURRENCY, prefix='cu_')
    buro_credit_type_dum = pd.get_dummies(buro.CREDIT_TYPE, prefix='ty_')

    buro_full = pd.concat(
        [
            buro, buro_credit_active_dum, buro_credit_currency_dum,
            buro_credit_type_dum
        ],
        axis=1)

    del buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum
    gc.collect()

    print('Merge with buro avg')
    buro_full = buro_full.merge(
        right=avg_buro_bal.reset_index(),
        how='left',
        on='SK_ID_BUREAU',
        suffixes=('', '_bur_bal'))

    print('Counting buro per SK_ID_CURR')
    nb_bureau_per_curr = buro_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
    buro_full['SK_ID_BUREAU'] = buro_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])

    print('Averaging bureau')
    avg_buro = buro_full.groupby('SK_ID_CURR').mean()
    print(avg_buro.head())

    del buro, buro_full
    gc.collect()

    # previous_application
    print('Read prev')
    prev = pd.read_csv(path + 'previous_application.csv')

    prev_cat_features = [
        f_ for f_ in prev.columns if prev[f_].dtype == 'object'
    ]

    print('Go to dummies')
    prev_dum = pd.DataFrame()
    for f_ in prev_cat_features:
        prev_dum = pd.concat(
            [prev_dum, pd.get_dummies(prev[f_], prefix=f_).astype(np.uint8)],
            axis=1)

    prev = pd.concat([prev, prev_dum], axis=1)

    del prev_dum
    gc.collect()

    print('Counting number of Prevs')
    nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])

    print('Averaging prev')
    avg_prev = prev.groupby('SK_ID_CURR').mean()
    print(avg_prev.head())
    del prev
    gc.collect()

    # POS_CASH_balance
    print('Reading POS_CASH')
    pos = pd.read_csv(path + 'POS_CASH_balance.csv')

    print('Go to dummies')
    pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'])], axis=1)

    print('Compute nb of prevs per curr')
    nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    print('Go to averages')
    avg_pos = pos.groupby('SK_ID_CURR').mean()

    del pos, nb_prevs
    gc.collect()

    print('Reading CC balance')
    cc_bal = pd.read_csv(path + 'credit_card_balance.csv')

    print('Go to dummies')
    cc_bal = pd.concat(
        [
            cc_bal, pd.get_dummies(
                cc_bal['NAME_CONTRACT_STATUS'], prefix='cc_bal_status_')
        ],
        axis=1)

    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    print('Compute average')
    avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
    avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]

    del cc_bal, nb_prevs
    gc.collect()

    # installments_payments
    print('Reading Installments')
    inst = pd.read_csv(path + 'installments_payments.csv')

    nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

    avg_inst = inst.groupby('SK_ID_CURR').mean()
    avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]


    return avg_buro, avg_prev, avg_pos, avg_cc_bal, avg_inst


def data_prep(df):

    # Filling NaN values with 0
    df = df.fillna(0)

    # Define categorical feats
    categorical_feats = [f for f in df.columns if df[f].dtype == 'object']
    for f_ in categorical_feats:
        df[f_], indexer = pd.factorize(df[f_])

    # Merge with add_data
    df = df.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    df = df.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    df = df.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    df = df.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    df = df.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

    # Normalize headers
    df = df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    return df

def data_prep_predict(df, avg_buro, avg_prev, avg_pos, avg_cc_bal, avg_inst, cols_2_keep):

    # Filling NaN values with 0
    df = df.fillna(0)

    # Define categorical feats
    categorical_feats = [f for f in df.columns if df[f].dtype == 'object']
    for f_ in categorical_feats:
        df[f_], indexer = pd.factorize(df[f_])

    # Merge with add_data
    df = df.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    df = df.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    df = df.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    df = df.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    df = df.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

    # Normalize headers
    df = df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    
    # Checking the presence of each column (for prediction)
    for col in cols_2_keep:
        if col not in df.columns:
            df[col] = 0

    return df[cols_2_keep]


def train_model(data_, y_, folds_):

    oof_preds = np.zeros(data_.shape[0])
    gof_preds = np.zeros(data_.shape[0])

    feature_importance_df = pd.DataFrame()

    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]

    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_.iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_.iloc[val_idx], y_.iloc[val_idx]
        print(trn_x)
        print(trn_y)
        # Applying combined random oversampling and undersampling for imbalanced data
        print(Counter(trn_y))
        # define oversampling strategy
        over = RandomOverSampler(sampling_strategy=0.2)
        # fit and apply the transform
        trn_x, trn_y = over.fit_resample(trn_x, trn_y)
        # summarize class distribution
        print(Counter(trn_y))
        # define undersampling strategy
        under = RandomUnderSampler(sampling_strategy=0.7)
        # fit and apply the transform
        trn_x, trn_y = under.fit_resample(trn_x, trn_y)
        # summarize class distribution
        print(Counter(trn_y))
        # define new data id
        new_ids = pd.Series.append(trn_x['SK_ID_CURR'], val_x['SK_ID_CURR'])
        print(new_ids)
        print(new_ids.shape)

        trn_x, trn_y = trn_x[feats], trn_y
        val_x, val_y = val_x[feats], val_y

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.03,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(
            trn_x,
            trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric='aucpr',
            verbose=100,
            early_stopping_rounds=100  
        )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        gof_preds[val_idx] = clf.predict(val_x, num_iteration=clf.best_iteration_)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' %
              (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        
      
        del trn_x, trn_y, val_x
        gc.collect()


    # oof full prediction evaluation
    #print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))

    # gof prediction evaluation
    # Metrics
    print(gof_preds.shape)
    print('Accuracy:', accuracy_score(val_y, gof_preds[val_idx]))
    print('F1 score:', f1_score(val_y, gof_preds[val_idx]))
    print('Recall:', recall_score(val_y, gof_preds[val_idx]))
    print('Precision:', precision_score(val_y, gof_preds[val_idx]))
    
    # Confusion matrix
    cf_matrix = confusion_matrix(val_y, gof_preds[val_idx])
    plt.figure(figsize=(6, 6))
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.title('Confusion matrix')
    plt.tight_layout()
    plt.savefig(path[:-12]+'graphs\\confusion_matrix.png')


    # Full confusion matrix
    cf_matrix = confusion_matrix(y, gof_preds)
    plt.figure(figsize=(6, 6))
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.title('Full confusion matrix')
    plt.tight_layout()
    plt.savefig(path[:-12]+'graphs\\confusion_matrix_full.png')


    # Classification report
    report = classification_report(val_y, gof_preds[val_idx], target_names=['0', '1'], output_dict=True)
    df_report = pd.DataFrame(report).T.reset_index()
    df_report = df_report.rename(columns={'f1-score':'f1score', 'index':''})
    print(df_report)
    columns = list(df_report.columns)
    values=[df_report.iloc[:,0]]
    for i in columns[1:5]:
        values.append(round(df_report[i], 2)) 
    fig = go.Figure(data=[go.Table(
        header=dict(values=columns,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=values,
                fill_color='lavender',
                align='left'))
    ])
    fig.write_image(path[:-12]+'graphs\\classification_report.png')

    # df compilation for export
    df_oof_preds = pd.DataFrame({'SK_ID_CURR':ids, 'PREDICTION':oof_preds, 'TARGET':y})
    df_oof_preds = df_oof_preds[['SK_ID_CURR', 'PREDICTION', 'TARGET']]
    
    df_gof_preds = pd.DataFrame({'SK_ID_CURR':ids, 'PREDICTION':gof_preds, 'TARGET':y})
    df_gof_preds = df_gof_preds[['SK_ID_CURR', 'PREDICTION', 'TARGET']]
    
    return clf, oof_preds, df_oof_preds, df_gof_preds, feature_importance_df, roc_auc_score(y, oof_preds), df_report, new_ids


def display_importances(feature_importance_df_):

    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby(
        "feature").mean().sort_values(
            by="importance", ascending=False)[:30].index

    best_features = feature_importance_df_.loc[
        feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(path[:-12]+'graphs\\lgbm_importances-01.png')
    return best_features

def display_roc_curve(y_, oof_preds_, folds_idx_):

    # Plot ROC curves
    plt.figure(figsize=(6, 6))
    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    plt.plot(
        [0, 1], [0, 1],
        linestyle='--',
        lw=2,
        color='r',
        label='Luck',
        alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(
        fpr,
        tpr,
        color='b',
        label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
        lw=2,
        alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(path[:-12]+'graphs\\roc_curve-01.png')


def display_precision_recall(y_, oof_preds_, folds_idx_):

    # Plot ROC curves
    plt.figure(figsize=(6, 6))

    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(
        precision,
        recall,
        color='b',
        label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
        lw=2,
        alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('LightGBM Recall / Precision')
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig(path[:-12]+'graphs\\recall_precision_curve-01.png')


if __name__ == '__main__':
    gc.enable()
    
    # Build model inputs
    avg_buro, avg_prev, avg_pos, avg_cc_bal, avg_inst = build_model_input()
    print('Read data')
    data = pd.read_csv(path + 'application_train.csv')
    print('Shape data: ', data.shape)
    ids_origin = data['SK_ID_CURR']

    # Columns for dashboard
    df_data_db = data[['SK_ID_CURR','CODE_GENDER', 'NAME_FAMILY_STATUS']]
    df_data_db = df_data_db.rename(columns={'CODE_GENDER':'Gender', 'NAME_FAMILY_STATUS':'Family_status'})
    df_data_db['Age'] = abs(round(data['DAYS_BIRTH']/365))
    print('Shape data for dashboard: ', df_data_db.shape)

    # Defining columns to keep
    y = data['TARGET']
    data = data.drop(columns=['TARGET'])
    print('Shape data: ', data.shape)

    # Processed data
    ids = data['SK_ID_CURR']
    data = data_prep(data)
    print('Shape data: ', data.shape)
    cols_2_keep = data.columns

    
    # Training and prediction for application_train
    # Create Folds
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)
    # Train model and get oof predictions
    clf, oof_preds, df_oof_preds, df_gof_preds, importances, score, df_report, new_ids = train_model(data, y, folds)
    # Save the model to disk
    pickle.dump(clf, open(path[:-12]+'src\\scoring_model' + '.sav', 'wb'))
    # Save train data predictions
    now = datetime.now()
    score = str(round(score, 6)).replace('.', '')
    oof_file = 'train_score_' + score + '_' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
    df_oof_preds.to_csv(oof_file, index=False)
    
    # Prepare application_test data for prediction
    print('Read test data')
    test = pd.read_csv(path + 'application_test.csv')
    print('Shape test data: ', test.shape)

    # Columns for dashboard
    df_test_db = test[['SK_ID_CURR','CODE_GENDER', 'NAME_FAMILY_STATUS']]
    df_test_db = df_test_db.rename(columns={'CODE_GENDER':'Gender', 'NAME_FAMILY_STATUS':'Family_status'})
    df_test_db['Age'] = abs(round(test['DAYS_BIRTH']/365))
    print('Shape data for dashboard: ', df_test_db.shape)

    ids_test = test['SK_ID_CURR']
    test = data_prep_predict(test, avg_buro, avg_prev, avg_pos, avg_cc_bal, avg_inst, cols_2_keep)
    print('Shape test data: ', test.shape)
    
    # Save application_test data predictions
    sub_preds = np.zeros(test.shape[0])
    feats = [f for f in test.columns if f not in ['SK_ID_CURR']]
    sub_preds += clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1]
    sub_preds_target = np.zeros(test.shape[0])
    sub_preds_target += clf.predict(test[feats], num_iteration=clf.best_iteration_)
    test['PREDICTION'] = sub_preds
    test['TARGET'] = sub_preds_target
    test['SK_ID_CURR'] = ids_test
    test_preds = test[['SK_ID_CURR', 'PREDICTION', 'TARGET']]
    now = datetime.now()
    test_file = 'test_score' + '_' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
    test_preds.to_csv(test_file, index=False)


    # Display a few graphs
    folds_idx = [(trn_idx, val_idx)
                 for trn_idx, val_idx in folds.split(data, y)]
    best_features = display_importances(feature_importance_df_=importances)
    pickle.dump(best_features, open(path[:-12]+'src\\features_importances.sav', 'wb'))

    display_roc_curve(y_=y, oof_preds_=oof_preds, folds_idx_=folds_idx)
    display_precision_recall(y_=y, oof_preds_=oof_preds, folds_idx_=folds_idx)


    # Save processed data
    PROCESSED_DATA_PATH = path[:-12]+'src\\data_processed' + '.csv'
    sampled_data = df_oof_preds[df_oof_preds.SK_ID_CURR.isin(new_ids)]
    sampled_data = sampled_data.merge(data, how='left', on='SK_ID_CURR')
    sampled_data = sampled_data.merge(df_data_db, how='left', on='SK_ID_CURR')
    print('sampled_data', sampled_data.shape)
    test = test.merge(df_test_db, how='left', on='SK_ID_CURR')
    print('test', test.shape)
    processed_data = pd.concat([test, sampled_data])
    processed_data = processed_data.sample(n=25000, random_state=1, axis=0)
    print('processed_data', processed_data.shape)
    processed_data.to_csv(PROCESSED_DATA_PATH, index=False)
    pickle.dump(cols_2_keep, open(path[:-12]+'src\\cols_2_keep.sav', 'wb'))