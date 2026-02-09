#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
このスクリプトは、指定CSV（相対パス時はout配下優先）を前処理し、処理後CSVを保存します。

preprocess_csv.py
指定したCSVを前処理して、同じフォルダに別名で保存するスクリプト。

使い方:
  python preprocess_csv.py "C:\\path\\to\\input.csv"
  python preprocess_csv.py "./input.csv" --suffix "_processed"
  python preprocess_csv.py "./input.csv" --encoding cp932
  python preprocess_csv.py "./input.csv" --dropna

依存:
  pip install pandas jeraconv
"""

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

BASE_DIR = Path("@localhost/public_html/yasusho-topics.com/wp-content/themes/cocoon-child-master")
OUT_DIR = BASE_DIR / "out"

# ====== CSVの列名に合わせて調整する場所 ======
COL_BUILD_YEAR = "建築年"
COL_TRADE_TIME = "取引時点"          # 例: "2020年第1四半期" など（4桁年を抽出）
COL_PRICE = "取引価格（総額）"

FEATURES = [
    "最寄駅：距離（分）",
    "間口",
    "面積（㎡）",
    "延床面積（㎡）",
    "前面道路：幅員（ｍ）",
    "容積率（％）",
]
# ===========================================


def _to_numeric_series(s: pd.Series) -> pd.Series:
    """数字/小数/マイナス以外を削って数値化（単位やカンマ混在を想定）"""
    cleaned = s.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def convert_wareki_to_seireki(series: pd.Series) -> pd.Series:
    """
    和暦(平成/令和/昭和) → 西暦(数値)
    対応例:
      平成28年 -> 2016
      令和2年  -> 2020
      昭和55年 -> 1980
      2001年   -> 2001
      不詳     -> NaN
    """

    def _convert(x):
        if pd.isna(x):
            return float("nan")

        s = str(x).strip()
        if s in {"", "不詳", "不明", "-", "—"}:
            return float("nan")

        # 西暦が含まれる場合
        m = re.search(r"(\d{4})", s)
        if m:
            year = int(m.group(1))
            if 1800 <= year <= 2100:
                return float(year)

        # 和暦
        m = re.search(r"(昭和|平成|令和)\s*(\d+)", s)
        if not m:
            return float("nan")

        era, year = m.group(1), int(m.group(2))

        if era == "昭和":   # 1926年 = 昭和1年
            return float(1925 + year)
        if era == "平成":   # 1989年 = 平成1年
            return float(1988 + year)
        if era == "令和":   # 2019年 = 令和1年
            return float(2018 + year)

        return float("nan")

    return series.apply(_convert).astype(float)



def extract_trade_year(series: pd.Series) -> pd.Series:
    """取引時点から4桁年を抽出して数値化。抽出できない場合はNaN。"""
    s = series.astype(str)
    year = s.str.extract(r"(\d{4})")[0]
    return pd.to_numeric(year, errors="coerce").astype(float)


def preprocess_df(df: pd.DataFrame, dropna: bool = False) -> pd.DataFrame:
    dfp = df.copy()

    # 建築年: 和暦 -> 西暦
    if COL_BUILD_YEAR in dfp.columns:
        dfp["建築年_西暦"] = convert_wareki_to_seireki(dfp[COL_BUILD_YEAR])
    else:
        dfp["建築年_西暦"] = float("nan")

    # 取引年抽出
    if COL_TRADE_TIME in dfp.columns:
        dfp["取引年"] = extract_trade_year(dfp[COL_TRADE_TIME])
    else:
        dfp["取引年"] = float("nan")

    # 築年数: 取引年 - 建築年 + 1
    mask = dfp["取引年"].notna() & dfp["建築年_西暦"].notna()
    dfp.loc[mask, "築年数"] = dfp.loc[mask, "取引年"] - dfp.loc[mask, "建築年_西暦"] + 1
    dfp.loc[~mask, "築年数"] = float("nan")

    # 価格の数値化
    if COL_PRICE in dfp.columns:
        dfp[COL_PRICE] = _to_numeric_series(dfp[COL_PRICE])
    else:
        dfp[COL_PRICE] = float("nan")

    # 特徴量も数値化
    for c in FEATURES:
        if c in dfp.columns:
            dfp[c] = _to_numeric_series(dfp[c])

    use_cols = [c for c in FEATURES if c in dfp.columns] + ["築年数", COL_PRICE, "建築年_西暦", "取引年"]
    out = dfp[use_cols].copy()

    if dropna:
        out = out.dropna(axis=0, how="any").reset_index(drop=True)

    return out


def read_csv_safely(path: Path, encoding: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """encoding指定がなければ utf-8-sig→cp932→utf-8 を順に試す。"""
    if encoding:
        return pd.read_csv(path, encoding=encoding), encoding

    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError:
            continue

    # ここには通常来ない想定
    return pd.read_csv(path, encoding="utf-8"), "utf-8"


def main() -> None:
    ap = argparse.ArgumentParser(description="特定CSVを前処理して同フォルダに別名保存")
    ap.add_argument("input_csv", help="入力CSVパス")
    ap.add_argument("--suffix", default="_processed", help="出力ファイル名の末尾（拡張子前）")
    ap.add_argument("--encoding", default=None, help="入力CSVのエンコーディング（例: utf-8-sig / cp932）")
    ap.add_argument("--dropna", action="store_true", help="欠損を含む行を削除する")
    args = ap.parse_args()

    in_path = Path(args.input_csv)
    if not in_path.is_absolute() and not in_path.exists():
        candidate = OUT_DIR / in_path
        if candidate.exists():
            in_path = candidate

    if not in_path.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {in_path}")

    df, used_enc = read_csv_safely(in_path, args.encoding)
    processed = preprocess_df(df, dropna=args.dropna)

    out_path = in_path.with_name(f"{in_path.stem}{args.suffix}{in_path.suffix}")
    processed.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"OK: {in_path.name} -> {out_path.name}")
    print(f"read encoding: {used_enc}")
    print(f"rows: {len(df)} -> {len(processed)} / cols: {processed.shape[1]}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
