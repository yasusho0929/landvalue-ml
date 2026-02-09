"""このスクリプトは、out配下の学習用CSVでランダムフォレスト回帰を実行し、予測結果CSVをout配下へ保存します。"""

from pathlib import Path

BASE_DIR = Path("@localhost/public_html/yasusho-topics.com/wp-content/themes/cocoon-child-master")
OUT_DIR = BASE_DIR / "out"

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def main():
    # ====== 設定 ======
    CSV_PATH = OUT_DIR / "testL_processed2.csv"
    TARGET = "PRICE_PER_TSUBO"
    OUTPUT_PATH = OUT_DIR / "testL_processed_forest_result.csv"

    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # ランダムフォレスト設定（まずは無難な値）
    N_ESTIMATORS = 500
    MAX_DEPTH = None
    MIN_SAMPLES_LEAF = 1
    N_JOBS = -1
    # ==================

    # 読み込み
    df = pd.read_csv(CSV_PATH)
    print("Loaded:", df.shape)

    # 目的変数チェック
    if TARGET not in df.columns:
        raise ValueError(f"TARGET列が見つかりません: {TARGET}")

    # 目的変数を数値化（念のため）
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()

    # 説明変数 / 目的変数
    DROP_COLS = ["VALUE"]  # 強リーク対策（必要なら増やす）
    drop_cols = [c for c in DROP_COLS if c in df.columns]

    X = df.drop(columns=[TARGET] + drop_cols)
    y = df[TARGET]

    # 数値/カテゴリ列判定
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print("Num cols:", len(num_cols))
    print("Cat cols:", len(cat_cols))

    # 前処理（RFはスケーリング不要。カテゴリだけOne-Hot）
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    # モデル（ランダムフォレスト）
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])

    # 分割（評価用）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 学習
    pipe.fit(X_train, y_train)

    # 評価
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:,.0f} (円/坪)")
    print(f"R2 : {r2:.3f}")

    # ==========================
    # 全行予測 → 列追加 → 出力
    # ==========================
    pred_all = pipe.predict(X)

    df_out = df.copy()
    df_out["PRED_PRICE_PER_TSUBO"] = pred_all

    # 相違率（予測−実測）/実測
    # 実測が0のケースがあれば無限大になるので保護
    denom = df_out[TARGET].replace(0, np.nan)
    df_out["DIFF_RATE"] = (df_out["PRED_PRICE_PER_TSUBO"] - df_out[TARGET]) / denom

    # CSV出力
    df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUTPUT_PATH}")

    # 先頭確認
    print(df_out[[TARGET, "PRED_PRICE_PER_TSUBO", "DIFF_RATE"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
