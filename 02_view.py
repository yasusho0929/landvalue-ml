"""このスクリプトは、out配下の前処理済みCSVを読み込み、先頭行と欠損状況を確認します。"""

import pandas as pd
from IPython.display import display
from pathlib import Path

BASE_DIR = Path("@localhost/public_html/yasusho-topics.com/wp-content/themes/cocoon-child-master")
OUT_DIR = BASE_DIR / "out"

df = pd.read_csv(OUT_DIR / "testL_processed.csv", encoding='utf-8-sig')
display(df.head(3))

total = df.isnull().sum()
percent = (df.isnull().sum()/df.isnull().count()*100)
missing_df  = pd.concat([total, percent], axis=1, keys=['欠損のあるレコード数', '欠損の全体に占める割合'])
display(missing_df)
