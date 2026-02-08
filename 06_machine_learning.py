import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score



def main():
    # ====== 設定 ======
    CSV_PATH = "testL_processed2.csv"   # 必要ならパスを変更
    TARGET = "PRICE_PER_TSUBO"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    ALPHA = 0.01
    # ==================

    # 読み込み
    df = pd.read_csv(CSV_PATH)
    print("Loaded:", df.shape)

    # 目的変数チェック
    if TARGET not in df.columns:
        raise ValueError(f"TARGET列が見つかりません: {TARGET}")

    # 目的変数を数値化（念のため）
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])

    # 説明変数 / 目的変数
    DROP_COLS = ["VALUE"]

    X = df.drop(columns=[TARGET] + DROP_COLS)
    y = df[TARGET]

    # 数値/カテゴリ列判定
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print("Num cols:", len(num_cols))
    print("Cat cols:", len(cat_cols))

    # 前処理
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # モデル
    model = Lasso(alpha=ALPHA, max_iter=10000)

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    # 分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 学習
    pipe.fit(X_train, y_train)

    # 予測 & 評価
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f} (円/坪)")
    print(f"R2 : {r2:.3f}")

    # 係数を見る（解釈用）
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    coef = pipe.named_steps["model"].coef_

    coef_df = pd.DataFrame({"feature": feature_names, "coef": coef})
    coef_df = coef_df.sort_values("coef", ascending=False)

    print("\n=== coef top 20 ===")
    print(coef_df.head(20).to_string(index=False))

    print("\n=== coef bottom 20 ===")
    print(coef_df.tail(20).to_string(index=False))

    # サンプル予測（例：先頭行）
    sample = X.iloc[[0]]
    pred = pipe.predict(sample)[0]
    print(f"\nSample prediction (row0): {pred:.1f} 円/坪")

    print(df["PRICE_PER_TSUBO"].describe())
    print("max:", df["PRICE_PER_TSUBO"].max())

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )
    plt.xlabel("Actual PRICE_PER_TSUBO (円/坪)")
    plt.ylabel("Predicted PRICE_PER_TSUBO (円/坪)")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
