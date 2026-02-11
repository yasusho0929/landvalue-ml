"""このスクリプトは、out配下の予測結果CSVの符号化列を名称に復元し、ラベル付きCSVをout配下へ保存します。"""


from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

# =========================
# 入出力ファイル
# =========================
INPUT_CSV = SCRIPT_DIR / "testLO_processed_forest_result.csv"
OUTPUT_CSV = SCRIPT_DIR / "testLO_processed_forest_result_labeled.csv"

PATHS = {
    "district": ROOT_DIR / "appendix/GifuIchiLatLng_deduplicated.csv",
    "station": ROOT_DIR / "appendix/駅の利用者.csv",
    "land_shape": ROOT_DIR / "appendix/土地形状別_坪単価平均.csv",
    "road_type": ROOT_DIR / "appendix/前面道路種類別_坪単価平均.csv",
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

# 地区（緯度・経度 -> 大字町丁目名）
def norm_coord(series):
    """座標を小数点以下6桁で正規化し、逆引きに使いやすい形にする。"""
    return pd.to_numeric(series, errors="coerce").round(6)


district_lookup = {
    (lat, lng): name
    for lat, lng, name in zip(
        norm_coord(df_district["緯度"]),
        norm_coord(df_district["経度"]),
        df_district["大字町丁目名"],
    )
    if not np.isnan(lat) and not np.isnan(lng)
}

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


def detect_lat_lng_columns(frame):
    """予測結果CSVの緯度経度カラム名を検出する。"""
    lat_candidates = ["lat", "LAT", "latitude", "LATITUDE", "緯度"]
    lng_candidates = ["lng", "LNG", "lon", "LON", "longitude", "LONGITUDE", "経度"]

    lat_col = next((c for c in lat_candidates if c in frame.columns), None)
    lng_col = next((c for c in lng_candidates if c in frame.columns), None)

    return lat_col, lng_col


def restore_district_from_coords(frame, lookup):
    lat_col, lng_col = detect_lat_lng_columns(frame)
    if lat_col is None or lng_col is None:
        print("[WARN] DISTRICT: 緯度経度カラムが見つからないため復元をスキップします")
        return pd.Series([np.nan] * len(frame), index=frame.index)

    lat_vals = norm_coord(frame[lat_col])
    lng_vals = norm_coord(frame[lng_col])

    restored = pd.Series(
        [lookup.get((lat, lng)) for lat, lng in zip(lat_vals, lng_vals)],
        index=frame.index,
    )

    miss = restored.isna().sum()
    if miss > 0:
        print(f"[WARN] DISTRICT: 復元できなかった行数 = {miss}")

    return restored


df["DISTRICT_NAME"] = restore_district_from_coords(df, district_lookup)
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
