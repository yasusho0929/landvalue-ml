"""このスクリプトは、out配下の前処理済みCSVを読み込み、基本統計と欠損・相関を確認します。"""

from pathlib import Path

BASE_DIR = Path("@localhost/public_html/yasusho-topics.com/wp-content/themes/cocoon-child-master")
OUT_DIR = BASE_DIR / "out"

# ==============================
# 0. import
# ==============================
from IPython.display import display
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# 表示設定（見やすく）
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


# ==============================
# 1. データ読み込み
# ==============================
# 適宜パスを変更
df = pd.read_csv(OUT_DIR / "merged_all_processed.csv", encoding="utf-8-sig")

print("=== shape ===")
print(df.shape)

print("\n=== head ===")
display(df.head())

print("\n=== info ===")
df.info()

print("\n=== describe ===")
display(df.describe(include="all"))


# ==============================
# 2. 欠損値チェック
# ==============================
print("\n=== 欠損値（件数） ===")
display(df.isnull().sum().sort_values(ascending=False))

print("\n=== 欠損値（割合 %） ===")
display((df.isnull().mean() * 100).round(2).sort_values(ascending=False))


# ==============================
# 3. 数値分布チェック（ヒストグラム）
# ==============================
numeric_cols = df.select_dtypes(include=np.number).columns

df[numeric_cols].hist(
    bins=50,
    figsize=(15, 10)
)
plt.tight_layout()
plt.show()


# ==============================
# 4. 外れ値チェック（箱ひげ図）
# ==============================
plt.figure(figsize=(15, 6))
df[numeric_cols].boxplot(rot=45)
plt.tight_layout()
plt.show()


# ==============================
# 5. カテゴリ変数チェック
# ==============================
categorical_cols = df.select_dtypes(exclude=np.number).columns

for col in categorical_cols:
    print(f"\n=== {col} value_counts (top10) ===")
    display(df[col].value_counts(dropna=False).head(10))


# ==============================
# 6. 相関チェック（価格がある場合）
# ==============================
TARGET_COL = "price"  # ← 目的変数名を変更

if TARGET_COL in df.columns:
    corr = df.corr(numeric_only=True)
    print("\n=== price 相関 ===")
    display(corr[TARGET_COL].sort_values(ascending=False))


# ==============================
# 7. 前処理後データを想定した簡易モデルチェック
# ==============================
# ※ one-hot 済み or 数値のみ前提
if TARGET_COL in df.columns:
    df_model = df.dropna()  # 簡易チェックなので一旦 dropna

    X = df_model.drop(columns=[TARGET_COL])
    y = df_model[TARGET_COL]

    # 数値のみ抽出（カテゴリ未処理でも落ちないように）
    X = X.select_dtypes(include=np.number)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Lasso(alpha=0.01)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Lasso R2（前処理健全性チェック用） ===")
    print(r2)

    print("\n=== 係数（重要特徴） ===")
    coef = pd.Series(model.coef_, index=X.columns)
    display(coef.sort_values(key=abs, ascending=False).head(20))


# ==============================
# 8. 前処理前後比較用（before / after がある場合）
# ==============================
# before_df.describe()
# after_df.describe()
