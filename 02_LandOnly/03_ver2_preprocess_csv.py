"""このスクリプトは、out配下の元データCSVを前処理し、学習用CSVをout配下へ出力します。"""

from pathlib import Path

import pandas as pd
import re

PROJECT_DIR = Path("@localhost/public_html/yasusho-topics.com/wp-content/themes/cocoon-child-master")
BASE_DIR = PROJECT_DIR / "02_LandOnly"
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 設定 =====
INPUT_CSV = OUT_DIR / "testL.csv"
OUTPUT_CSV = OUT_DIR / "merged_all_processed.csv"



# ★重要：このCSVはCP932
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
print("read:", df.shape)

# ===== 1) 住宅地のみ =====
# 値ブレ対策：NFKC正規化してから比較
df["地域"] = df["地域"].astype(str).str.normalize("NFKC")
df = df[df["地域"] == "住宅地"]
print("after 地域==住宅地:", df.shape)

# ===== 2) 不要列削除（あなたのヘッダに合わせる）=====
drop_cols = [
    "用途", "地域", "種類", "坪単価", "取引価格（㎡単価）",
    "土地の形状", "間口", "取引の事情等", "今後の利用目的",
    "前面道路：種類",
    # 任意（学習に使わないなら落としてOK）
    "価格情報区分",
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
print("after drop cols:", df.shape)

# ===== 3) 必須列の欠損を落とす（取引時期に対応）=====
required_cols = [
    "建築年", "最寄駅：距離（分）", "延床面積（㎡）", "都市計画",
    "最寄駅：名称", "建物の構造", "建ぺい率（％）", "容積率（％）",
    "前面道路：幅員（ｍ）", "取引時期",
    "面積（㎡）", "取引価格（総額）",
]
df = df.dropna(subset=required_cols)
print("after dropna required:", df.shape)

# ===== 4) 文字列の正規化 =====
for col in ["取引時期", "都市計画", "建築年"]:
    df[col] = df[col].astype(str).str.normalize("NFKC")

# ===== 5) 建築年：和暦→西暦 =====
df = df[df["建築年"] != "戦前"]
df.loc[df["建築年"] == "平成元年", "建築年"] = "平成1年"

df["年号"] = df["建築年"].str[:2]
df["和暦_年"] = pd.to_numeric(df["建築年"].str[2:].str.replace("年", "", regex=False), errors="coerce")

df.loc[df["年号"] == "昭和", "西暦_年"] = df["和暦_年"] + 1925
df.loc[df["年号"] == "平成", "西暦_年"] = df["和暦_年"] + 1988

df = df.dropna(subset=["西暦_年"])
df["西暦_年"] = df["西暦_年"].astype(int)

df = df.drop(columns=["建築年", "年号", "和暦_年"])
print("after buildyear->CY:", df.shape)

# ===== 6) 取引時期：年 & 四半期 =====
# 例: 2023年第4四半期
df["TDY"] = pd.to_numeric(df["取引時期"].str.extract(r"^(\d{4})")[0], errors="coerce")
df["取引時期_四半期"] = pd.to_numeric(df["取引時期"].str.extract(r"第(\d)四半期")[0], errors="coerce")
df = df.dropna(subset=["TDY", "取引時期_四半期"])
df["TDY"] = df["TDY"].astype(int)
df["取引時期_四半期"] = df["取引時期_四半期"].astype(int)

df["TDQ"] = df["TDY"].astype(str) + "0" + df["取引時期_四半期"].astype(str)
df["AGE"] = df["TDY"] - df["西暦_年"]
print("after TDY/TDQ/AGE:", df.shape)

# ===== 7) 面積・延床：2000㎡以上の除外 & 数値化 =====
df = df[df["面積（㎡）"] != "2000㎡以上"]
df = df[df["延床面積（㎡）"] != "2000㎡以上"]

df["面積（㎡）"] = pd.to_numeric(df["面積（㎡）"], errors="coerce")
df["延床面積（㎡）"] = pd.to_numeric(df["延床面積（㎡）"], errors="coerce")
df = df.dropna(subset=["面積（㎡）", "延床面積（㎡）"])

# LANDはint化（小数が出るならここを消してfloatのままでもOK）
df["面積（㎡）"] = df["面積（㎡）"].astype(int)

# ===== 8) 駅距離：変なカテゴリ除外 & 数値化 =====
exclude_time = ["1H30?2H", "1H?1H30", "30分?60分", "2H?"]
df = df[~df["最寄駅：距離（分）"].isin(exclude_time)]
df["最寄駅：距離（分）"] = pd.to_numeric(df["最寄駅：距離（分）"], errors="coerce")
df = df.dropna(subset=["最寄駅：距離（分）"])
df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].astype(int)

print("after LAND/FLOOR/TIME clean:", df.shape)

# ===== 9) rename（あなたの仕様 + 取引時期対応）=====
df = df.rename(columns={
    "取引価格（総額）": "PRICE",
    "延床面積（㎡）": "FLOOR",
    "最寄駅：距離（分）": "TIME",
    "面積（㎡）": "LAND",
    "建ぺい率（％）": "BLR",
    "容積率（％）": "FRA",
    "市区町村コード": "CITY_CODE",
    "西暦_年": "CY",
    "前面道路：幅員（ｍ）": "ROAD_WIDTH",
    "都道府県名": "ADDRESS1",
    "市区町村名": "ADDRESS2",
    "地区名": "ADDRESS3",
    "建物の構造": "STRUCTURE",
    "都市計画": "USE",
    "前面道路：方位": "DIRECTION",
    "最寄駅：名称": "STATION",
    "取引時期": "TD_STR",
})

# ===== 10) 保存 =====
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print("saved:", OUTPUT_CSV, df.shape)

total = df.isnull().sum()
percent = (df.isnull().sum() / df.isnull().count() * 100)
missing_df  = pd.concat([total, percent], axis=1, keys=['欠損のあるレコード数', '欠損の全体に占める割合'])

try:
    display(missing_df)
except NameError:
    print(missing_df.sort_values('欠損のあるレコード数', ascending=False))
