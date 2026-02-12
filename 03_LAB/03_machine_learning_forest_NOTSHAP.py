"""このスクリプトは、学習用CSVでランダムフォレスト回帰を実行し、予測結果CSVを保存します。

実行の前提パス:
@localhost/public_html/yasusho-topics.com/wp-content/themes/cocoon-child-master/

このファイルを最初に実行したあとに「入力」する内容（ターミナルで順に実行するコマンド）:
1) まずこのスクリプトを実行
   python3 02_LO/04.5_machine_learning_forest_NOTSHAP.py

2) 予測結果にラベル名を戻す
   python3 02_LO/05_restore_labels.py

3) 地区ごとの割高/割安ランキングを表示
   python3 02_LO/06_district_over_under_rank.py

補足:
- 対話入力（キーボードで値を入れる処理）はありません。
- 必要に応じて 04 の CSV_PATH を、前段の出力CSV（通常: testLO_processed.csv）に合わせてください。
"""


from pathlib import Path
import json
import time
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = "./models/rf_landprice.joblib"
COLUMNS_PATH = "./models/rf_landprice_columns.json"


def align_features(X: pd.DataFrame, saved_columns: list[str]) -> pd.DataFrame:
    """学習時のカラム順に合わせる（不足列は0補完、余剰列は削除）。"""
    X_aligned = X.copy()

    missing_cols = [c for c in saved_columns if c not in X_aligned.columns]
    if missing_cols:
        print(f"[INFO] 不足カラムを0補完します: {len(missing_cols)}列")
        for col in missing_cols:
            X_aligned[col] = 0

    extra_cols = [c for c in X_aligned.columns if c not in saved_columns]
    if extra_cols:
        print(f"[INFO] 余分なカラムを削除します: {len(extra_cols)}列")

    return X_aligned.reindex(columns=saved_columns, fill_value=0)


def main():
    # ====== 設定 ======
    CSV_PATH = SCRIPT_DIR / "testLAB_processed.csv"
    TARGET = "BUILDING_VALUE"
    OUTPUT_PATH = SCRIPT_DIR / "testLAB_processed_forest_result.csv"

    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # ランダムフォレスト設定（まずは無難な値）
    N_ESTIMATORS = 500
    MAX_DEPTH = None
    MIN_SAMPLES_LEAF = 1
    N_JOBS = -1
    # ==================

    model_path = SCRIPT_DIR / "models" / "rf_building_value.joblib"
    columns_path = SCRIPT_DIR / "models" / "rf_building_value_columns.json"

    # 読み込み
    df = pd.read_csv(CSV_PATH)
    print("Loaded:", df.shape)

    # 目的変数チェック
    if TARGET not in df.columns:
        raise ValueError(f"TARGET列が見つかりません: {TARGET}")

    # 目的変数を数値化（念のため）
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()

    # BUILDING_VALUE外れ値分析・除外（IQR法）
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        target_valid = df[TARGET].dropna()
        if not target_valid.empty:
            q1 = target_valid.quantile(0.25)
            q3 = target_valid.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (df[TARGET] < lower) | (df[TARGET] > upper)
            outlier_count = int(outlier_mask.sum())
            print(
                f"[INFO] {TARGET}外れ値分析(IQR): Q1={q1:.3e}, Q3={q3:.3e}, "
                f"Lower={lower:.3e}, Upper={upper:.3e}, 除外件数={outlier_count}"
            )
            df = df.loc[~outlier_mask].copy()
            print(f"[INFO] {TARGET}外れ値除外後のデータ件数: {df.shape[0]}")

    # 説明変数 / 目的変数
    # STRUCTURE は学習しない（文字列で冗長）
    DROP_COLS = ["STRUCTURE"]  # 強リーク対策（必要なら増やす）
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

    # モデル読込/学習
    columns_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists() and columns_path.exists():
        print(f"[INFO] 学習済みモデルを読み込みます: {model_path}")
        pipe = joblib.load(model_path)
        with columns_path.open("r", encoding="utf-8") as f:
            saved_columns = json.load(f)
        print(f"[INFO] 学習時カラム情報を読み込みました: {columns_path}")

        X_train = align_features(X_train, saved_columns)
        X_test = align_features(X_test, saved_columns)
        X = align_features(X, saved_columns)
    else:
        print("[INFO] 学習済みモデルがないため新規学習を実行します。")
        pipe.fit(X_train, y_train)
        saved_columns = X_train.columns.tolist()

        joblib.dump(pipe, model_path)
        with columns_path.open("w", encoding="utf-8") as f:
            json.dump(saved_columns, f, ensure_ascii=False, indent=2)

        print(f"[INFO] モデルを保存しました: {model_path}")
        print(f"[INFO] 学習時カラム情報を保存しました: {columns_path}")

    # 評価
    print("[INFO] テストデータの予測を開始します...")
    t0 = time.perf_counter()
    y_pred = pipe.predict(X_test)
    print(f"[INFO] テストデータ予測が完了しました（{time.perf_counter() - t0:.1f}秒）")

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:,.0f} (円)")
    print(f"R2 : {r2:.3f}")

    # ==========================
    # 全行予測 → 列追加 → 出力
    # ==========================
    print("[INFO] 全行データの予測を開始します...")
    t0 = time.perf_counter()
    pred_all = pipe.predict(X)
    print(f"[INFO] 全行データの予測が完了しました（{time.perf_counter() - t0:.1f}秒）")

    df_out = df.copy()
    df_out["PRED_BUILDING_VALUE"] = pred_all

    # 相違率（予測−実測）/実測
    # 実測が0のケースがあれば無限大になるので保護
    denom = df_out[TARGET].replace(0, np.nan)
    df_out["DIFF_RATE"] = (df_out["PRED_BUILDING_VALUE"] - df_out[TARGET]) / denom

    # CSV出力
    print("[INFO] CSV出力を開始します...")
    t0 = time.perf_counter()
    df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] CSV出力が完了しました（{time.perf_counter() - t0:.1f}秒）")
    print(f"\nSaved: {OUTPUT_PATH}")

    # 先頭確認
    print(df_out[[TARGET, "PRED_BUILDING_VALUE", "DIFF_RATE"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
