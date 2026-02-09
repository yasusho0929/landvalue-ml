"""このスクリプトは、out配下の予測結果CSVの符号化列を名称に復元し、ラベル付きCSVをout配下へ保存します。"""

from pathlib import Path

BASE_DIR = Path("@localhost/public_html/yasusho-topics.com/wp-content/themes/cocoon-child-master")
OUT_DIR = BASE_DIR / "out"

import pandas as pd
import numpy as np

# =========================
# 入出力ファイル
# =========================
INPUT_CSV = OUT_DIR / "testL_processed_forest_result.csv"
OUTPUT_CSV = OUT_DIR / "testL_processed_forest_result_labeled.csv"

PATHS = {
    "district": OUT_DIR / "地区名別_坪単価平均.csv",
    "station": OUT_DIR / "駅の利用者.csv",
    "land_shape": OUT_DIR / "土地形状別_坪単価平均.csv",
    "road_type": OUT_DIR / "前面道路種類別_坪単価平均.csv",
}

# =========================
# 読み込み
# =========================
df = pd.read_csv(INPUT_CSV)
print("Loaded result CSV:", df.shape)

# 対応表CSV読み込み
df_district = pd.read_csv(PATHS["district"])
df_station = pd.read_csv(PATHS["station"])
df_land_shape = pd.read_csv(PATHS["land_shape"])
df_road_type = pd.read_csv(PATHS["road_type"])

# =========================
# 逆引き辞書を作成
# （数値 → 名称）
# =========================

# 地区
district_rev = dict(
    zip(df_district["坪単価"], df_district["地区名"])
)

# 駅（利用者数 → 駅名）
station_rev = dict(
    zip(df_station["利用者数"], df_station["駅名"])
)

# 土地形状
land_shape_rev = dict(
    zip(df_land_shape["坪単価"], df_land_shape["土地の形状"])
)

# 前面道路種類
road_type_rev = dict(
    zip(df_road_type["坪単価"], df_road_type["前面道路：種類"])
)

# =========================
# 復元（数値 → 文字）
# =========================

def restore_col(series, rev_map, col_name):
    restored = series.map(rev_map)

    # マッチしなかったものを確認
    miss = restored.isna().sum()
    if miss > 0:
        print(f"[WARN] {col_name}: 復元できなかった行数 = {miss}")

    return restored


df["DISTRICT_NAME"] = restore_col(df["DISTRICT"], district_rev, "DISTRICT")
df["NEAREST_STATION_NAME"] = restore_col(df["NEAREST_STATION"], station_rev, "NEAREST_STATION")
df["LAND_SHAPE_NAME"] = restore_col(df["LAND_SHAPE"], land_shape_rev, "LAND_SHAPE")
df["ROAD_TYPE_NAME"] = restore_col(df["ROAD_TYPE"], road_type_rev, "ROAD_TYPE")

# =========================
# 保存
# =========================
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("Saved:", OUTPUT_CSV)

# 確認
print(df[[
    "DISTRICT", "DISTRICT_NAME",
    "NEAREST_STATION", "NEAREST_STATION_NAME",
    "LAND_SHAPE", "LAND_SHAPE_NAME",
    "ROAD_TYPE", "ROAD_TYPE_NAME",
]].head())
