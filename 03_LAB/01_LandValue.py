#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""建物付き土地データに対して土地価格・建物価格を推定するスクリプト。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

INPUT_CSV = ROOT_DIR / "01_LandOnlyAndLandAndBuilding/merged_land_and_building.csv"
MODEL_PATH = ROOT_DIR / "02_LO/models/rf_landprice.joblib"
COLUMNS_PATH = ROOT_DIR / "02_LO/models/rf_landprice_columns.json"
OUTPUT_CSV = SCRIPT_DIR / "01_BuildingValue.csv"


def read_csv_robust(path: Path) -> pd.DataFrame:
    enc_candidates = ["utf-8-sig", "cp932", "utf-8"]
    last_err = None
    for enc in enc_candidates:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"CSVを読めませんでした: {path}") from last_err


def normalize_str(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "　": pd.NA})


def to_numeric_series(s: pd.Series) -> pd.Series:
    s = normalize_str(s)
    s = s.str.replace(",", "", regex=False)
    s = s.str.translate(str.maketrans("０１２３４５６７８９．－", "0123456789.-"))
    return pd.to_numeric(s, errors="coerce")


def parse_minutes_from_station_distance(x: str | pd.NA) -> float | np.nan:
    if x is pd.NA or x is None:
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan

    if re.fullmatch(r"\d+(\.\d+)?", s):
        try:
            return float(s)
        except Exception:  # noqa: BLE001
            return np.nan

    s = s.replace("分", "")
    s = s.replace("〜", "～").replace("~", "～")
    s = s.upper().replace("Ｈ", "H")

    def token_to_min(tok: str) -> float | None:
        tok = tok.strip()
        if tok == "":
            return None
        m = re.fullmatch(r"(\d+)\s*H\s*(\d+)?", tok)
        if m:
            h = int(m.group(1))
            mm = int(m.group(2)) if m.group(2) else 0
            return float(h * 60 + mm)
        if re.fullmatch(r"\d+(\.\d+)?", tok):
            return float(tok)
        return None

    if "～" in s:
        a, b = s.split("～", 1)
        amin = token_to_min(a)
        bmin = token_to_min(b)
        if amin is not None and bmin is not None:
            return (amin + bmin) / 2.0
        if amin is not None:
            return float(amin)
        if bmin is not None:
            return float(bmin)
        return np.nan

    v = token_to_min(s)
    return float(v) if v is not None else np.nan


def parse_transaction_year(s: pd.Series) -> pd.Series:
    y = normalize_str(s).str.extract(r"(\d{4})\s*年", expand=False)
    return pd.to_numeric(y, errors="coerce")


def prepare_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 文字列列を正規化
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = normalize_str(df[col])

    # 数値列
    numeric_cols = [
        "面積（㎡）",
        "間口",
        "前面道路：幅員（ｍ）",
        "建ぺい率（％）",
        "容積率（％）",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_numeric_series(df[col])

    # 距離（分）
    df["STATION_DISTANCE_MIN"] = df["最寄駅：距離（分）"].apply(parse_minutes_from_station_distance)

    # 取引年
    df["TRANSACTION_YEAR"] = parse_transaction_year(df["取引時期"])

    # 最寄駅：名称（括弧以降削除）
    df["最寄駅：名称"] = (
        df["最寄駅：名称"]
        .astype("string")
        .str.replace(r"[（(].*$", "", regex=True)
        .str.strip()
        .replace({"": pd.NA})
    )

    # 最寄駅：名称 -> 利用者数
    station_df = read_csv_robust(ROOT_DIR / "appendix/駅の利用者.csv")
    station_map = dict(zip(station_df.iloc[:, 0], station_df.iloc[:, 1]))

    # 前面道路：種類 -> 坪単価平均
    road_type_df = read_csv_robust(ROOT_DIR / "appendix/前面道路種類別_坪単価平均.csv")
    road_type_map = dict(zip(road_type_df.iloc[:, 0], road_type_df.iloc[:, 1]))

    # 土地の形状 -> 坪単価平均
    land_shape_df = read_csv_robust(ROOT_DIR / "appendix/土地形状別_坪単価平均.csv")
    land_shape_map = dict(zip(land_shape_df.iloc[:, 0], land_shape_df.iloc[:, 1]))

    # 地区名 -> 緯度経度
    district_df = read_csv_robust(ROOT_DIR / "appendix/GifuIchiLatLng_deduplicated.csv")
    district_lat_lng = district_df[["大字町丁目名", "緯度", "経度"]].rename(
        columns={"大字町丁目名": "地区名", "緯度": "lat", "経度": "lng"}
    )

    # 方位を数値化
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

    df = df.merge(district_lat_lng, on="地区名", how="left")

    features = pd.DataFrame(
        {
            "NEAREST_STATION": df["最寄駅：名称"].map(station_map),
            "AREA_SQM": df["面積（㎡）"],
            "LAND_SHAPE": df["土地の形状"].map(land_shape_map),
            "FRONTAGE": df["間口"],
            "DIRECTION": df["前面道路：方位"].map(direction_map),
            "ROAD_TYPE": df["前面道路：種類"].map(road_type_map),
            "ROAD_WIDTH": df["前面道路：幅員（ｍ）"],
            "BUILDING_COVERAGE_RATIO": df["建ぺい率（％）"],
            "FLOOR_AREA_RATIO": df["容積率（％）"],
            "STATION_DISTANCE_MIN": df["STATION_DISTANCE_MIN"],
            "TRANSACTION_YEAR": df["TRANSACTION_YEAR"],
            "lat": pd.to_numeric(df["lat"], errors="coerce"),
            "lng": pd.to_numeric(df["lng"], errors="coerce"),
        }
    )

    # すべての数値列をfloat64に明示的に変換してpandas.NAを避ける
    numeric_cols_features = features.select_dtypes(include=["number"]).columns
    for col in numeric_cols_features:
        features[col] = features[col].astype("float64")

    return features


def align_features(X: pd.DataFrame, saved_columns: list[str]) -> pd.DataFrame:
    X_aligned = X.copy()

    missing_cols = [c for c in saved_columns if c not in X_aligned.columns]
    for col in missing_cols:
        X_aligned[col] = 0

    return X_aligned.reindex(columns=saved_columns, fill_value=0)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"学習済みモデルが見つかりません: {MODEL_PATH}")

    df_in = read_csv_robust(INPUT_CSV)

    features = prepare_features(df_in)

    with COLUMNS_PATH.open("r", encoding="utf-8") as f:
        saved_columns = json.load(f)

    features = align_features(features, saved_columns)

    model = joblib.load(MODEL_PATH)
    pred_price_per_tsubo = model.predict(features)

    df_out = df_in.copy()
    df_out["PRED_PRICE_PER_TSUBO"] = pred_price_per_tsubo

    area_sqm = to_numeric_series(df_out["面積（㎡）"])
    transaction_total = to_numeric_series(df_out["取引価格（総額）"])

    df_out["LAND_VALUE"] = df_out["PRED_PRICE_PER_TSUBO"] * area_sqm / 3.3058
    df_out["BUILDING_VALUE"] = transaction_total - df_out["LAND_VALUE"]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_CSV}")
    print(df_out[["PRED_PRICE_PER_TSUBO", "LAND_VALUE", "BUILDING_VALUE"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
