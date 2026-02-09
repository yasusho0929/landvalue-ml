#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
このスクリプトは、out配下の元データCSVを機械学習用に整形し、前処理済みCSVをout配下へ保存します。

preprocess_template.py (customized)

要件反映:
- 目的変数は y ではなく VALUE にする（= 取引価格（総額））
- 取引月 列は削除
- 今後の利用目的 列は削除
- 以下が空白/欠損の行は削除:
  地区名, 最寄駅：名称, 間口, 前面道路：方位, 前面道路：種類, 前面道路：幅員（ｍ）,
  都市計画, 建ぺい率（％）, 容積率（％）, 最寄駅：距離（分）_num
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path("@localhost/public_html/yasusho-topics.com/wp-content/themes/cocoon-child-master")
OUT_DIR = BASE_DIR / "out"

CSV_PATH = OUT_DIR / "testL.csv"
TARGET_COL = "取引価格（総額）"  # VALUE に入れる元


def read_csv_robust(path: Path) -> pd.DataFrame:
    enc_candidates = ["utf-8-sig", "cp932", "utf-8"]
    last_err = None
    for enc in enc_candidates:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    try:
        return pd.read_csv(path, encoding="cp932", errors="replace")
    except Exception:
        raise last_err


def normalize_str(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s = s.str.strip()
    # よくある「空白扱い」を NaN に
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "　": pd.NA})
    return s


def to_numeric_series(s: pd.Series) -> pd.Series:
    s = normalize_str(s)
    s = s.str.replace(",", "", regex=False)
    s = s.str.translate(str.maketrans("０１２３４５６７８９．－", "0123456789.-"))
    return pd.to_numeric(s, errors="coerce")


def parse_quarter_to_date(x: str | pd.NA) -> pd.Timestamp | pd.NaT:
    if x is pd.NA or x is None:
        return pd.NaT
    x = str(x).strip()
    m = re.search(r"(\d{4})\s*年第\s*([1-4])\s*四半期", x)
    if not m:
        return pd.NaT
    year = int(m.group(1))
    q = int(m.group(2))
    month_map = {1: 2, 2: 5, 3: 8, 4: 11}
    return pd.Timestamp(year=year, month=month_map[q], day=15)


def parse_minutes_from_station_distance(x: str | pd.NA) -> float | np.nan:
    if x is pd.NA or x is None:
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan

    if re.fullmatch(r"\d+(\.\d+)?", s):
        try:
            return float(s)
        except Exception:
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
            return h * 60 + mm
        if re.fullmatch(r"\d+(\.\d+)?", tok):
            return float(tok)
        return None

    if "～" in s:
        a, b = s.split("～", 1)
        amin = token_to_min(a)
        bmin = token_to_min(b)
        if amin is not None and bmin is not None:
            return (amin + bmin) / 2.0
        if amin is not None and bmin is None:
            return float(amin)
        if amin is None and bmin is not None:
            return float(bmin)
        return np.nan

    v = token_to_min(s)
    return float(v) if v is not None else np.nan


def preprocess_for_ml(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = df.copy()

    # 列名正規化
    df.columns = [str(c).strip().replace("\u3000", " ") for c in df.columns]

    # 文字列列を整形
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = normalize_str(df[col])

    # 数値化（必要に応じて増減）
    numeric_cols = [
        "取引価格（総額）",
        "面積（㎡）",
        "間口",
        "前面道路：幅員（ｍ）",
        "建ぺい率（％）",
        "容積率（％）",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_numeric_series(df[col])

    # 最寄駅距離（分）を数値化
    if "最寄駅：距離（分）" in df.columns:
        df["最寄駅：距離（分）_num"] = df["最寄駅：距離（分）"].apply(parse_minutes_from_station_distance)

    # 取引時期（四半期）→ 日付 → 年/月
    if "取引時期" in df.columns:
        df["取引時期_date"] = df["取引時期"].apply(parse_quarter_to_date)
        df["取引年"] = df["取引時期_date"].dt.year
        df["取引月"] = df["取引時期_date"].dt.month

    # ---- 要件: 行削除（空白/欠損チェック） ----
    must_have_cols = [
        "地区名",
        "最寄駅：名称",
        "間口",
        "面積（㎡）",          # ← 追加
        "土地の形状",          # ← 追加
        "前面道路：方位",
        "前面道路：種類",
        "前面道路：幅員（ｍ）",
        "都市計画",
        "建ぺい率（％）",
        "容積率（％）",
        "最寄駅：距離（分）_num",
        TARGET_COL,
    ]
    must_have_cols = [c for c in must_have_cols if c in df.columns]

    # --- 最寄駅：名称 の正規化（"(以降" を削除） ---
    if "最寄駅：名称" in df.columns:
        df["最寄駅：名称"] = (
            df["最寄駅：名称"]
            .astype("string")
            .str.replace(r"[（(].*$", "", regex=True)
            .str.strip()
            .replace({"": pd.NA})
        )


    # 文字列の必須列は normalize 済みなので NA を落とすだけでOK
    # 数値列も to_numeric_series 済みなので NA を落とすだけでOK
    df = df.dropna(subset=must_have_cols).copy()

    # ---- 要件: 列削除 ----
    drop_feature_cols = [
        "取引月",         # 作った列を削除
        "今後の利用目的",  # 削除指定
    ]
    drop_feature_cols = [c for c in drop_feature_cols if c in df.columns]
    df = df.drop(columns=drop_feature_cols)

    # ---- VALUE を作る（yではなくVALUE） ----
    if TARGET_COL not in df.columns:
        raise KeyError(f"TARGET_COL={TARGET_COL} が見つかりません。列名を確認してください。")

    df["VALUE"] = df[TARGET_COL]
    VALUE = df["VALUE"]

    # 学習に使わない列（必要なら調整）
    drop_cols_for_X = [
        TARGET_COL,
        "VALUE",
        "取引時期",        # 元文字列
        "取引時期_date",   # 日付そのものは残したければ外す
        "最寄駅：距離（分）",  # 元文字列
    ]
    drop_cols_for_X = [c for c in drop_cols_for_X if c in df.columns]

    X = df.drop(columns=drop_cols_for_X)

    # 欠損率が高すぎる列を削除（任意）
    na_rate = X.isna().mean()
    too_sparse = na_rate[na_rate > 0.80].index.tolist()
    if too_sparse:
        X = X.drop(columns=too_sparse)

    df_ml = pd.concat([X, VALUE.rename("VALUE")], axis=1)
    return df_ml, X, VALUE


def main() -> None:
    df = read_csv_robust(CSV_PATH)
    df_ml, X, VALUE = preprocess_for_ml(df)

    print("=== Loaded ===", df.shape)
    print("=== After preprocess ===", df_ml.shape)

    print("\n--- dtypes (X) ---")
    # あなたの環境では sort_values が通ったとのことなのでそのまま
    print(X.dtypes.astype(str).sort_values())

    print("\n--- VALUE describe ---")
    print(VALUE.describe())

    out_path = CSV_PATH.with_name(CSV_PATH.stem + "_processed.csv")
    df_ml.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
