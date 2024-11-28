import pandas as pd
import numpy as np
import statistics as st
import random
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
import os
  # 数値型列の統計量を表示

train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

train_test = pd.concat([train, test])


def missing_values(df, name):
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    # 欠損値がある列のみを抽出
    missing_data = missing_percentage[missing_percentage > 0]

    # 可視化
    if not missing_data.empty:
        plt.figure(figsize=(10, 6))
        missing_data.sort_values(ascending=False).plot(kind='bar')
        plt.title('欠損値の割合 (各カラム)', fontsize=16)
        plt.ylabel('欠損割合 (%)', fontsize=12)
        plt.xlabel('カラム', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("欠損値のある列はありません。")

    df.info()
df = train_test

def visulalize_data_info(data,path):
# 作業ディレクトリ内のoutputsフォルダパス
    base_dir = os.path.abspath("outputs")
    output_dir = os.path.join(base_dir, path)

    # outputsフォルダを作成（存在しない場合）
    os.makedirs(output_dir, exist_ok=True)

    # 数値型列の可視化（ヒストグラム）
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        plt.figure()
        sns.histplot(data[col], kde=True, bins=30, color='blue')
        plt.title(f'numerical - {col}', fontsize=16)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'{col}_histogram.png'))  # 保存先を指定
        plt.close()

    # カテゴリ型列の可視化（棒グラフ）
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        plt.figure()
        data[col].value_counts().plot(kind='bar', color='orange')
        plt.title(f'categorical - {col}', fontsize=16)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'{col}_bar_chart.png'))  # 保存先を指定
        plt.close()

    print(f"グラフが保存されました: {output_dir}")

    describe_result = data.describe(include='all')  # 数値とカテゴリを含む統計量

# ファイル出力先パス
# output_path_csv = os.path.join(output_dir, "data_description.csv")
    output_path_txt = os.path.join(output_dir, "data_description.txt")
    with open(output_path_txt, "w") as f:
        f.write(describe_result.to_string())   





missing_values(train,"train_null_data")
missing_values(test,"test null data")

visulalize_data_info(train,"train_visualize")
visulalize_data_info(test,"test_visualize")





