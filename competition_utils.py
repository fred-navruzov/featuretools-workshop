import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import time
import gc
import featuretools as ft


from os import cpu_count
from featuretools import selection
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
)

from lightgbm import LGBMClassifier

# PREPROCESSING BLOCK -------------------------------------------------------------------------------
def reduce_mem_usage(df, skip_cols_pattern='SK_ID_'):
    """ 
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:

        if skip_cols_pattern in col:
            print(f"don't optimize index {col}")

        else:
            col_type = df[col].dtype

            if col_type != object:

                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def clean_non_overlapped_values(df_train, df_test, fill_value=np.nan, concat=False, verbose=True):
    """
    Check for labels in test, not present in train and vice versa (XOR)
    and substitute them by selected value (default=np.nan)
    """
    useless_values = dict()

    for c in df_train.columns[df_train.dtypes == 'object']:
        c_train = set(df_train[c].unique())
        c_test = set(df_test[c].unique())
        
        diff = c_train ^ c_test # XOR

        if diff:
            if verbose:
                print(f'feature {c} has different values: {diff}')
            useless_values[c] = diff
    
    if verbose:
        print(useless_values)
    
    # substitute those by specified value
    for (c, val) in useless_values.items():
        df_train.loc[:, c] = df_train.loc[:, c].map(
            lambda x: x if x not in val else fill_value
        )
        
        df_test.loc[:, c] = df_test.loc[:, c].map(
            lambda x: x if x not in val else fill_value
        )
       
    if concat:
        return pd.concat([df_train, df_test])
    else:
        return df_train, df_test

def remove_li_features(df):
    """Remove low information features"""
    old_shape = df.shape[1]
    df = selection.remove_low_information_features(df)
    print('Removed features from df: {}'.format(old_shape - df.shape[1]))

    return df

def replace_day_outliers(df):
    """Replace 365243 with np.nan in any columns with DAYS"""
    
    for col in df.columns:
        if "DAYS" in col:
            df[col] = df[col].replace({365243: np.nan})

    return df

def replace_missing_app(df):
    """Deal with outliers/missing in APPLICATION table"""

    # replace officially defined NaNs
    df = df.replace(['XNA', 'XAP'], np.nan)

    # build null-df
    null_values_apptr = df.isnull().sum()
    null_values_apptr = null_values_apptr[null_values_apptr != 0]\
    .sort_values(ascending = False).reset_index() #only show rows with null values
    null_values_apptr.columns = ["variable", "n_missing"]
            
    # impute counts
    for variable in null_values_apptr["variable"]:
        
        # replace null mode/median blocks with 0
        if (variable.endswith("MEDI")|variable.endswith("AVG")):
            df.loc[:,variable] = df.loc[:,variable].fillna(0)
        
        elif (variable.startswith("AMT_REQ_CREDIT_BUREAU")):
            df.loc[:,variable] = df.loc[:,variable].fillna(0)
            
        elif (variable.endswith("SOCIAL_CIRCLE")):
            df.loc[:,variable] = df.loc[:,variable].fillna(0)
            
        elif (variable.startswith("AMT_REQ_CREDIT_BUREAU")):
            df.loc[:,variable] = df.loc[:,variable].fillna(0)
        
        if (variable.endswith("SOCIAL_CIRCLE")):
            df.loc[:,variable] = df.loc[:,variable].fillna(0)

    # dealing with external sources
    df.loc[:,"EXT_SOURCE_2"] = df.loc[:,"EXT_SOURCE_2"]\
    .fillna(df.EXT_SOURCE_2.median())

    # let's impute this by correspondent MODE
    for education in df.NAME_EDUCATION_TYPE.unique():
        mode_to_impute = df[
            df.NAME_EDUCATION_TYPE == education].OCCUPATION_TYPE.mode()[0]
        df.loc[
            df.NAME_EDUCATION_TYPE == education, "OCCUPATION_TYPE"
        ] = df.loc[df.NAME_EDUCATION_TYPE == education, "OCCUPATION_TYPE"]\
        .fillna(mode_to_impute)
        
    # name suite
    df.NAME_TYPE_SUITE = df.NAME_TYPE_SUITE.fillna(
        df.NAME_TYPE_SUITE.mode()[0])

    # family members
    df.CNT_FAM_MEMBERS = df.CNT_FAM_MEMBERS.fillna(df.CNT_FAM_MEMBERS.mode()[0])

    # last phone change

    df.DAYS_LAST_PHONE_CHANGE = \
    df.DAYS_LAST_PHONE_CHANGE.fillna(df.DAYS_LAST_PHONE_CHANGE.mode()[0])

    # annuity
    df.AMT_ANNUITY = df.AMT_ANNUITY.fillna(df.AMT_ANNUITY.median())

    # amt goods price
    df.AMT_GOODS_PRICE = df.AMT_GOODS_PRICE.fillna(df.AMT_GOODS_PRICE.mode()[0])

    # name suite
    df.NAME_TYPE_SUITE = df.NAME_TYPE_SUITE.fillna(df.NAME_TYPE_SUITE.mode()[0])

    return df

def categorical_encode(df, df_test=None, verbose=True):
    """Label-encode categorical features"""

    cat_columns = df.select_dtypes(include=['object']).columns

    for c in cat_columns:
        if verbose:
            print(f'Column: {c}...')
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype('str').fillna('missing'))
        if df_test is not None and c in df_test.columns:
            df_test[c] = le.transform(df_test[c].astype('str').fillna('missing'))

    if df_test is not None:
        return df, df_test 
    else:
        return df

def aligh_shape(df, df_test, target_col='TARGET'):
    """Intersect 2 dataframe's columns to align shape[1]"""
    train_labels = df[target_col]
    df, df_test = df.align(df_test, join='inner', axis=1)
    df[target_col] = train_labels

    print('Final training shape: ', df.shape)
    print('Final testing shape: ', df_test.shape)

    return df, df_test


# EVALUATION BLOCK ------------------------------------------------------------------------------
def make_clf(classifier='lightgbm', **params):
    """Creates classifier with given **params"""
    if classifier == 'lightgbm':
        return LGBMClassifier(**params)
    else:
        raise NotImplementedError('Sorry, haven\t implemented yet :)')

def train_model(data_, test_, y_, folds_, strategy='mean', model_params={}, 
    cols_to_exclude=['TARGET'], target_col='TARGET', folds=StratifiedKFold(), classifier='lightgbm'):
    """Train model in OOF-fashion and test averaging"""

    # define placeholders for oof predictions
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros((test_.shape[0], folds_.n_splits))
    
    # -//- for feature importance
    feature_importance_df = pd.DataFrame()
    
    feats = [f for f in data_.columns if f not in cols_to_exclude]
    
    # for each fold - define model, train it and compute oof
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        print('Fold #{}...'.format(n_fold + 1))

        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]    
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        
        # construct model with given type and params
        clf = make_clf(classifier, **model_params)
        # fit classifier

        if classifier == 'lightgbm':
            clf.fit(
                trn_x, trn_y, 
                eval_set=[(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', 
                verbose=25, 
                early_stopping_rounds=25,
                #categorical_feature='auto',
            )
            
            # calculate oof predictions
            oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        
            sub_preds[:, n_fold] = (
                clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] 
            )

        elif classifier == 'catboost':
            clf.fit(
                X=trn_x, 
                y=trn_y, 
                eval_set=[(val_x, val_y)], 
                #verbose_eval=50,
                use_best_model=True, 
            )

            # calculate oof predictions
            oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]
            
            sub_preds[:, n_fold] = (
                clf.predict_proba(test_[feats])[:, 1] 
            )
            
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
        
    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds)) 
    
    if strategy == 'mean':
        sub_preds_agg = sub_preds.mean(axis=1)
    elif strategy == 'median':
        sub_preds_agg = sub_preds.median(axis=1)
    else:
        sub_preds_agg = sub_preds.mean(axis=1)
        
    test_[target_col] = sub_preds_agg
    
    return (
        oof_preds, 
        #test_.reset_index()[['SK_ID_CURR', target_col]], 
        test_[[target_col]],
        sub_preds,  # raw fold predictions
        feature_importance_df
    )
    
def display_importances(feature_importance_df_, top_features=50):
    """Plot top-n feature importances of a classifier"""

    cols = feature_importance_df_[["feature", "importance"]]\
    .groupby("feature").mean().sort_values(
        by="importance", ascending=False
    )[:top_features].index
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize=(10, top_features // 5))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    #plt.tight_layout()

def display_roc_curve(y_, oof_preds_, folds_idx_):
    """Plot ROC-AUC curve and calculates AUC"""

    # Plot ROC curves
    plt.figure(figsize=(6,6))
    scores = [] 
    for n_fold, (_, val_idx) in enumerate(folds_idx_):  
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' \
                 % (n_fold + 1, score))
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(fpr, tpr, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()