import pandas as pd
import numpy as np
from difflib import get_close_matches

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score


# =========================
# 定数・設定
# =========================
TSUBO_TO_SQM = 3.3058  # 1坪 = 3.3058㎡

PATHS = {
    "train": "testL_processed2.csv",
    "district": "地区名別_坪単価平均.csv",
    "road_type": "前面道路種類別_坪単価平均.csv",
    "land_shape": "土地形状別_坪単価平均.csv",
    "station": "駅の利用者.csv",
}

TARGET = "PRICE_PER_TSUBO"
DROP_COLS = ["VALUE"]  # 強リーク防止

USE_COLS = [
    "DISTRICT",
    "NEAREST_STATION",
    "STATION_DISTANCE_MIN",
    "AREA_SQM",
    "FRONTAGE",
    "ROAD_WIDTH",
    "FLOOR_AREA_RATIO",
    "BUILDING_COVERAGE_RATIO",
    "LAND_SHAPE",
    "DIRECTION",
    "ROAD_TYPE",
    "TRANSACTION_YEAR",
]

# 方位対応（画像どおり）
DIRECTION_MAP = {
    "南": 1.0,
    "南東": 0.8,
    "南西": 0.8,
    "東": 0.6,
    "西": 0.6,
    "北東": 0.4,
    "北西": 0.4,
    "北": 0.2,
}


# =========================
# ユーティリティ
# =========================
def suggest_name(name, choices, n=5):
    return get_close_matches(name, list(choices), n=n, cutoff=0.6)





def load_maps(paths: dict):
    df_d = pd.read_csv(paths["district"])
    df_r = pd.read_csv(paths["road_type"])
    df_s = pd.read_csv(paths["land_shape"])
    df_st = pd.read_csv(paths["station"])

    # dict化（名称→数値）
    district_map = dict(zip(df_d["地区名"], df_d["坪単価"]))
    road_type_map = dict(zip(df_r["前面道路：種類"], df_r["坪単価"]))
    land_shape_map = dict(zip(df_s["土地の形状"], df_s["坪単価"]))
    station_map = dict(zip(df_st["駅名"], df_st["利用者数"]))

    return district_map, road_type_map, land_shape_map, station_map


def fit_slope_for_interpretation(df_train: pd.DataFrame, col: str):
    """
    col の値（加工特徴量）から、PRICE_PER_TSUBO（円/坪）の平均をざっくり推定するための比例係数を推定。
    例：地区特徴量(=DISTRICT) → 地区平均との差を円/坪で出すための換算に使う。
    """
    g = df_train.groupby(col)[TARGET].mean().reset_index()
    x = g[col].values.astype(float)
    y = g[TARGET].values.astype(float)
    denom = float(np.dot(x, x))
    if denom == 0:
        return 0.0
    a = float(np.dot(x, y) / denom)  # y ≈ a*x
    return a


def train_model(df: pd.DataFrame):
    # 目的変数を数値に
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()

    # 列選択
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=[TARGET] + drop)
    # USE_COLSで固定（安全）
    X = X[USE_COLS].copy()
    y = df[TARGET].copy()

    # 全部数値として扱う（今回のprocessed2はカテゴリが数値化済みの前提）
    num_cols = X.columns.tolist()
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    model = Lasso(alpha=0.01, max_iter=10000)

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    # 評価
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return pipe, (X_train, X_test, y_train, y_test, y_pred, mae, r2), df


def input_with_map(prompt, mapping: dict, label: str):
    while True:
        name = input(prompt).strip()
        if name in mapping:
            return name, mapping[name]
        sug = suggest_name(name, mapping.keys(), n=8)
        print(f"[{label}] 見つかりません: {name}")
        if sug:
            print("近い候補:", " / ".join(sug))
        print("もう一度入力してください。")


def input_float(prompt, min_val=None, max_val=None):
    while True:
        s = input(prompt).strip()
        try:
            v = float(s)
        except ValueError:
            print("数値で入力してください。")
            continue
        if min_val is not None and v < min_val:
            print(f"{min_val}以上で入力してください。")
            continue
        if max_val is not None and v > max_val:
            print(f"{max_val}以下で入力してください。")
            continue
        return v


def input_int(prompt, min_val=None, max_val=None):
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
        except ValueError:
            print("整数で入力してください。")
            continue
        if min_val is not None and v < min_val:
            print(f"{min_val}以上で入力してください。")
            continue
        if max_val is not None and v > max_val:
            print(f"{max_val}以下で入力してください。")
            continue
        return v


def count_comps(df_all: pd.DataFrame, district_val: float, station_users: int,
                station_dist: float, area_sqm: float):
    """
    簡易コンプ数（近傍事例数）：
    - DISTRICTが同一（同じ値）
    - 駅利用者数が±20%以内
    - 駅距離が±10分以内
    - 面積が±20%以内
    """
    df = df_all.copy()

    # 欠損対策
    for c in ["DISTRICT", "NEAREST_STATION", "STATION_DISTANCE_MIN", "AREA_SQM", TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["DISTRICT", "NEAREST_STATION", "STATION_DISTANCE_MIN", "AREA_SQM", TARGET])

    cond = (
        (df["DISTRICT"] == district_val) &
        (df["NEAREST_STATION"].between(station_users * 0.8, station_users * 1.2)) &
        (df["STATION_DISTANCE_MIN"].between(station_dist - 10, station_dist + 10)) &
        (df["AREA_SQM"].between(area_sqm * 0.8, area_sqm * 1.2))
    )
    return int(cond.sum())


def make_report(pipe, df_all, slopes, user_input_row: pd.DataFrame,
                district_name: str, station_name: str, road_type_name: str, land_shape_name: str):
    pred_yen = float(pipe.predict(user_input_row)[0])
    pred_man = pred_yen / 10000.0

    district_val = float(user_input_row["DISTRICT"].iloc[0])
    # 地区基準（円/坪）：DISTRICT特徴量→円/坪に換算（学習データから推定した比例係数）
    district_base_yen = slopes["DISTRICT"] * district_val
    diff_vs_district = pred_yen - district_base_yen

    comps = count_comps(
        df_all=df_all,
        district_val=district_val,
        station_users=int(user_input_row["NEAREST_STATION"].iloc[0]),
        station_dist=float(user_input_row["STATION_DISTANCE_MIN"].iloc[0]),
        area_sqm=float(user_input_row["AREA_SQM"].iloc[0]),
    )

    # コメント
    if diff_vs_district > 15000:
        judge = "やや割高（地区基準より +1.5万円/坪 以上）"
    elif diff_vs_district < -15000:
        judge = "割安寄り（地区基準より -1.5万円/坪 以下）"
    else:
        judge = "相場圏内（地区基準±1.5万円/坪）"

    lines = []
    lines.append("===================================")
    lines.append("  1物件評価レポート（PRICE_PER_TSUBO）")
    lines.append("===================================")
    lines.append("")
    lines.append("[入力（名称系）]")
    lines.append(f"- 地区名: {district_name}")
    lines.append(f"- 最寄駅: {station_name}")
    lines.append(f"- 前面道路種類: {road_type_name}")
    lines.append(f"- 土地の形状: {land_shape_name}")
    lines.append("")
    lines.append("[入力（数値系）]")
    cols_show = ["STATION_DISTANCE_MIN", "AREA_SQM", "FRONTAGE", "ROAD_WIDTH",
                 "BUILDING_COVERAGE_RATIO", "FLOOR_AREA_RATIO", "DIRECTION", "TRANSACTION_YEAR"]
    for c in cols_show:
        lines.append(f"- {c}: {user_input_row[c].iloc[0]}")
    lines.append("")
    lines.append("[予測結果]")
    lines.append(f"- 予測坪単価: {pred_yen:,.0f} 円/坪（{pred_man:,.2f} 万円/坪）")
    lines.append("")
    lines.append("[基準比較（地区水準ベース）]")
    lines.append(f"- 地区基準（推定）: {district_base_yen:,.0f} 円/坪")
    lines.append(f"- 地区平均との差: {diff_vs_district:+,.0f} 円/坪")
    lines.append(f"- 判定: {judge}")
    lines.append("")
    lines.append("[近傍事例（簡易）]")
    lines.append(f"- 近傍条件（地区一致/駅力±20%/駅距離±10分/面積±20%）に合う件数: {comps} 件")
    lines.append("")
    lines.append("※注意: 本レポートは「意思決定補助（多少リークOK）」前提の推定です。")
    lines.append("===================================")
    return "\n".join(lines)


def main():
    print("=== START 06_machine_learning.py ===")

    # マップ読み込み
    district_map, road_type_map, land_shape_map, station_map = load_maps(PATHS)

    # 学習データ読み込み＆学習
    df = pd.read_csv(PATHS["train"])
    pipe, eval_pack, df_all = train_model(df)

    X_train, X_test, y_train, y_test, y_pred, mae, r2 = eval_pack

    print("\n[モデル評価（参考）]")
    print(f"MAE: {mae:,.0f} 円/坪（{mae/10000:.2f} 万円/坪）")
    print(f"R2 : {r2:.3f}")

    # 解釈用の換算係数（レポートの地区基準に使う）
    slopes = {
        "DISTRICT": fit_slope_for_interpretation(df_all, "DISTRICT"),
    }

    print("\n[入力開始] 名称はCSVにある表記で入力してください。")

    # 名称入力→数値に変換
    district_name, district_val = input_with_map("地区名を入力: ", district_map, "地区名")
    station_name, station_users = input_with_map("最寄駅名を入力: ", station_map, "駅名")
    road_type_name, road_type_val = input_with_map("前面道路種類を入力: ", road_type_map, "前面道路種類")
    land_shape_name, land_shape_val = input_with_map("土地の形状を入力: ", land_shape_map, "土地の形状")

    # 数値入力
    station_dist = input_float("最寄駅までの距離（分）: ", min_val=0)
    area_sqm = input_float("面積（㎡）: ", min_val=1)
    frontage = input_float("間口（m）: ", min_val=0)
    road_width = input_float("前面道路幅員（m）: ", min_val=0)
    bcr = input_float("建ぺい率（%）: ", min_val=0, max_val=100)
    far = input_float("容積率（%）: ", min_val=0)
    direction = input_float("前面道路：方位（数値）: ")  # processed2に合わせた数値入力
    year = input_int("取引年（例: 2025）: ", min_val=1900, max_val=2100)

    # 1行データ作成（モデル入力形式に揃える）
    row = pd.DataFrame([{
        "DISTRICT": float(district_val),
        "NEAREST_STATION": int(station_users),
        "STATION_DISTANCE_MIN": float(station_dist),
        "AREA_SQM": float(area_sqm),
        "FRONTAGE": float(frontage),
        "ROAD_WIDTH": float(road_width),
        "FLOOR_AREA_RATIO": float(far),
        "BUILDING_COVERAGE_RATIO": float(bcr),
        "LAND_SHAPE": float(land_shape_val),
        "DIRECTION": float(direction),
        "ROAD_TYPE": float(road_type_val),
        "TRANSACTION_YEAR": int(year),
    }])[USE_COLS]

    report = make_report(
        pipe=pipe,
        df_all=df_all,
        slopes=slopes,
        user_input_row=row,
        district_name=district_name,
        station_name=station_name,
        road_type_name=road_type_name,
        land_shape_name=land_shape_name
    )

    print("\n" + report)


if __name__ == "__main__":
    main()
