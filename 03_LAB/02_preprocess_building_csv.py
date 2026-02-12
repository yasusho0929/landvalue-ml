#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""01_BuildingValue.csvを処理してtestLAB_processed.csvにする"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "01_BuildingValue.csv"
OUTPUT_CSV = SCRIPT_DIR / "testLAB_processed.csv"


def read_csv_robust(path: Path) -> pd.DataFrame:
    """複数のエンコーディングを試してCSVを読み込む"""
    enc_candidates = ["utf-8-sig", "cp932", "utf-8"]
    last_err = None
    for enc in enc_candidates:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSVを読めませんでした: {path}") from last_err


def normalize_str(s: pd.Series) -> pd.Series:
    """文字列列を正規化"""
    s = s.astype("string").str.strip()
    return s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "　": pd.NA})


def to_numeric(s: pd.Series) -> pd.Series:
    """文字列を数値に変換"""
    s = normalize_str(s)
    # 年、ヶ月など単位を削除
    s = s.str.replace(r"[年ヶ月日]", "", regex=True)
    s = s.str.replace(",", "", regex=False)
    s = s.str.translate(str.maketrans("０１２３４５６７８９．－", "0123456789.-"))
    return pd.to_numeric(s, errors="coerce")


def parse_year_from_trading_period(s: str) -> int | float:
    """取引時期から年を抽出 (例: "2021年第1四半期" -> 2021)"""
    if pd.isna(s) or s == "":
        return np.nan
    s = str(s).strip()
    m = re.search(r"(\d{4})\s*年", s)
    if m:
        return int(m.group(1))
    return np.nan


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {INPUT_CSV}")

    df = read_csv_robust(INPUT_CSV)
    
    print(f"入力行数: {len(df)}")

    # 削除対象の列
    drop_cols = [
        "地区名",
        "最寄駅：名称",
        "最寄駅：距離（分）",
        "取引価格（総額）",
        "面積（㎡）",
        "土地の形状",
        "間口",
        "延床面積（㎡）",
        "建築年",
        "建物の構造",
        "前面道路：方位",
        "前面道路：種類",
        "前面道路：幅員（ｍ）",
        "建ぺい率（％）",
        "容積率（％）",
        "取引時期",
    ]

    # 削除対象の列を正規化
    for col in drop_cols:
        if col in df.columns:
            df[col] = normalize_str(df[col])

    # これらの列のいずれかが欠損している行を削除
    df = df.dropna(subset=drop_cols, how="any")
    
    print(f"削除後の行数: {len(df)}")

    # 新しい列を作成
    result = pd.DataFrame()

    # YEAR = 取引時期 - 建築年
    trading_years = df["取引時期"].apply(parse_year_from_trading_period)
    building_years = to_numeric(df["建築年"])
    result["YEAR"] = trading_years - building_years

    # STRUCTURE = 建物の構造
    structure_str = normalize_str(df["建物の構造"])
    result["STRUCTURE"] = structure_str

    # ダミー変数（建物の構造に基づく）
    result["RC"] = (structure_str.str.contains("RC", case=False, na=False)).astype(int)
    result["MOKU"] = (structure_str.str.contains("木", case=False, na=False)).astype(int)
    result["TETSU"] = (structure_str.str.contains("鉄", case=False, na=False)).astype(int)
    result["LIGHT_TETSU"] = (structure_str.str.contains("軽量鉄骨", case=False, na=False)).astype(int)
    result["BLK"] = (structure_str.str.contains("ブロック", case=False, na=False)).astype(int)

    # 面積（㎡）
    result["面積（㎡）"] = to_numeric(df["面積（㎡）"])

    # TotalArea = 延床面積（㎡）
    result["TotalArea"] = to_numeric(df["延床面積（㎡）"])

    # BUILDING_VALUE
    result["BUILDING_VALUE"] = df["BUILDING_VALUE"]

    # NaNを含む行を削除（各数値列が変換に失敗した場合）
    result = result.dropna(how="any")

    print(f"最終行数: {len(result)}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_CSV}")
    print("\nFirst 10 rows:")
    print(result.head(10).to_string())


if __name__ == "__main__":
    main()
