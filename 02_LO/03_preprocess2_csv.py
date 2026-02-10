"""このスクリプトは、out配下の前処理CSVを追加加工し、学習用特徴量CSVをout配下へ保存します。"""


from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

# 入力ファイル
input_path = SCRIPT_DIR / "testL_processed.csv"
output_path = SCRIPT_DIR / "testL_processed2.csv"

# CSV読み込み（文字化け対策：utf-8-sig）
df = pd.read_csv(input_path, encoding="utf-8-sig")

# リネーム
df = df.rename(columns={
    "前面道路：方位": "DIRECTION"
})

# 削除する列
drop_cols = [
    "種類",
    "価格情報区分",
    "地域",
    "市区町村コード",
    "都道府県名",
    "市区町村名",
    "取引価格（㎡単価）",
    "都市計画",
]

# 実際に存在する列だけ削除（安全策）
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 方角の数値マッピング
direction_map = {
    "南": 1.0,
    "南東": 0.8,
    "南西": 0.8,
    "東": 0.6,
    "西": 0.6,
    "北東": 0.4,
    "北西": 0.4,
    "北": 0.2,
}

# 方角列名（※必要なら変更）
col_direction = "DIRECTION"

# 変換
df[col_direction] = df[col_direction].map(direction_map)


rename_map = {
    "地区名": "DISTRICT",
    "最寄駅：名称": "NEAREST_STATION",
    "坪単価": "PRICE_PER_TSUBO",
    "面積（㎡）": "AREA_SQM",
    "土地の形状": "LAND_SHAPE",
    "間口": "FRONTAGE",
    "前面道路：種類": "ROAD_TYPE",
    "前面道路：幅員（ｍ）": "ROAD_WIDTH",
    "建ぺい率（％）": "BUILDING_COVERAGE_RATIO",
    "容積率（％）": "FLOOR_AREA_RATIO",
    "最寄駅：距離（分）_num": "STATION_DISTANCE_MIN",
    "取引年": "TRANSACTION_YEAR",
}



# 存在する列だけリネーム
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


# =========================
# 外部CSVを使った数値化
# =========================

# ① 最寄駅：名称 → 利用者数
station_df = pd.read_csv(ROOT_DIR / "appendix/駅の利用者.csv", encoding="utf-8-sig")

# 想定カラム例：
# 駅名, 利用者数
station_map = dict(zip(
    station_df["駅名"],
    station_df["利用者数"]
))

df["NEAREST_STATION"] = df["NEAREST_STATION"].map(station_map)


# ② 前面道路：種類 → 坪単価平均
road_type_df = pd.read_csv(ROOT_DIR / "appendix/前面道路種類別_坪単価平均.csv", encoding="utf-8-sig")

# 想定カラム例：
# 前面道路：種類, 坪単価平均
road_type_map = dict(zip(
    road_type_df.iloc[:, 0],
    road_type_df.iloc[:, 1]
))

df["ROAD_TYPE"] = df["ROAD_TYPE"].map(road_type_map)


# ③ 地区名 → 坪単価平均
district_df = pd.read_csv(ROOT_DIR / "appendix/地区名別_坪単価平均.csv", encoding="utf-8-sig")

district_map = dict(zip(
    district_df.iloc[:, 0],
    district_df.iloc[:, 1]
))

df["DISTRICT"] = df["DISTRICT"].map(district_map)


# ④ 土地の形状 → 坪単価平均
land_shape_df = pd.read_csv(ROOT_DIR / "appendix/土地形状別_坪単価平均.csv", encoding="utf-8-sig")

land_shape_map = dict(zip(
    land_shape_df.iloc[:, 0],
    land_shape_df.iloc[:, 1]
))

df["LAND_SHAPE"] = df["LAND_SHAPE"].map(land_shape_map)


# =========================
# 変換後チェック
# =========================
print("カテゴリ変換後の欠損数:")
print(df[[
    "NEAREST_STATION",
    "ROAD_TYPE",
    "DISTRICT",
    "LAND_SHAPE"
]].isna().sum())


# 出力
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("出力完了:", output_path)
print("残っている列一覧:")
print(df.columns.tolist())
