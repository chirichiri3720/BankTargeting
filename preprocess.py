import pandas as pd
import numpy as np
import statistics as st
import random
from itertools import combinations

train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

train_test = pd.concat([train, test])

def missing_value_checker(df, name):
    chk_null = df.isnull().sum()
    chk_null_pct = chk_null / (df.index.max() + 1)
    chk_null_tbl = pd.concat([chk_null[chk_null > 0], chk_null_pct[chk_null_pct > 0]], axis=1)
    chk_null_tbl = chk_null_tbl.rename(columns={0: "欠損数",1: "欠損割合"})
    print(name)
    print(chk_null_tbl, end="\n\n")

def drop_columns(df):
    target_columns = [
        "default",
    ]

    return df.drop(target_columns,axis=1)

def categorical_encoding(df):
    target_columns = [
        'campaign',
    ]

    df['campaign_1'] = ((df['campaign'] == 1)).astype(int)
    df['campaign_2'] = ((df['campaign'] == 2)).astype(int)
    df['campaign_3'] = ((df['campaign'] == 3)).astype(int)
    df['campaign_4over'] = ((df['campaign'] >= 4)).astype(int)


    df = df.drop(target_columns,axis=1)

    return df

def calculate_std(x, y):
    return pd.concat([x, y], axis=1).std(axis=1, ddof=1)

def calculate_mean(x, y):
    return (x + y) / 2

def calculate_median(x, y):
    return pd.concat([x, y], axis=1).median(axis=1)

def calculate_q75(x, y):
    return pd.concat([x, y], axis=1).quantile(0.75, axis=1)

def calculate_q25(x, y):
    return pd.concat([x, y], axis=1).quantile(0.25, axis=1)

def calculate_zscore(x, mean, std):
    return (x - mean) / (std + 1e-3)

def conbination_columns(df):
    new_features = []
    feature_columns = df[['fico','revol.util']]
    for (feature1, feature2) in combinations(feature_columns, 2):
            f1, f2 = df[feature1], df[feature2]

            # 既存の特徴量操作
            new_features.append(f1 + f2)
            new_features.append(f1 - f2)
            new_features.append(f1 * f2)
            new_features.append(f1 / (f2 + 1e-8))
    
            # # 新しい特徴量操作
            # new_features.append(calculate_mean(f1, f2))
            # new_features.append(calculate_median(f1, f2))
            # new_features.append(calculate_q75(f1, f2))
            # new_features.append(calculate_q25(f1, f2))
            # zscore_f1 = calculate_zscore(
            #     f1, calculate_mean(f1, f2), calculate_std(f1, f2))
            # new_features.append(zscore_f1)
    new_features_df = pd.concat(new_features, axis=1)

    new_features_df.columns = [f'{feature1}_{operation}_{feature2}'
                                for (feature1, feature2) in combinations(feature_columns, 2)
                                # for operation in ['multiplied_by', 'divided_by', 'q75', 'q25', 'zscore_f1']]
                                # for operation in ['plus', 'minus', 'multiplied_by', 'divided_by', 'mean', 'median', 'q75', 'q25', 'zscore_f1']]
                                for operation in ['plus', 'minus', 'multiplied_by', 'divided_by']]
    
    result_df = pd.concat([df, new_features_df], axis=1)

    return result_df

def days_with_bin(df):
    target_columns = [
          'days.with.cr.line'
     ]
    
    df['days_0-2500'] = ((df['days.with.cr.line'] >= 0) & (df['days.with.cr.line'] < 2500)).astype(int)
    df['days_2500-5000'] = ((df['days.with.cr.line'] >= 2500) & (df['days.with.cr.line'] < 5000)).astype(int)
    df['days_5000-10000'] = ((df['days.with.cr.line'] >= 5000) & (df['days.with.cr.line'] < 10000)).astype(int)
    # df['days_10000-15000'] = ((df['days.with.cr.line'] >= 10000) & (df['days.with.cr.line'] < 15000)).astype(int)
    df['days_10000-'] = ((df['days.with.cr.line'] >= 10000)).astype(int)
    # df = df.drop(target_columns,axis=1)

    return df
   
df = train_test

df.to_csv('datasets/concat_fix.csv', index=False)

print(df.info())
train_test = df

drop_columns(df)

categorical_encoding(df)

train = train_test.iloc[:len(train)]
test = train_test.iloc[len(train):]

test = test.drop("y",axis=1)

# csvファイルの作成
train.to_csv('datasets/train_fix.csv', index=False)
test.to_csv('datasets/test_fix.csv', index=False)
