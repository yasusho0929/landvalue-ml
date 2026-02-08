import pandas as pd

# =========================
# 設定
# =========================
INPUT_CSV = "testL_processed_forest_result_labeled.csv"
MIN_COUNT = 30      # この件数未満の地区は除外
TOP_N = 20           # 上位何件出すか

# =========================
# 読み込み
# =========================
df = pd.read_csv(INPUT_CSV)

# 必須列チェック
required_cols = ["DISTRICT_NAME", "DIFF_RATE"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"必要な列がありません: {c}")

# =========================
# 地区別集計
# =========================
grp = (
    df
    .groupby("DISTRICT_NAME")
    .agg(
        mean_diff_rate=("DIFF_RATE", "mean"),
        median_diff_rate=("DIFF_RATE", "median"),
        count=("DIFF_RATE", "count"),
    )
    .reset_index()
)

# 件数フィルタ
grp = grp[grp["count"] >= MIN_COUNT].copy()

# =========================
# 割高率ワースト地区
# =========================
overpriced = grp.sort_values(
    "mean_diff_rate", ascending=False
).head(TOP_N)

# =========================
# 割安率ベスト地区
# =========================
underpriced = grp.sort_values(
    "mean_diff_rate", ascending=True
).head(TOP_N)

# =========================
# 表示
# =========================
print("===================================")
print(" 割高率ワースト地区（平均DIFF_RATE）")
print("===================================")
print(
    overpriced[[
        "DISTRICT_NAME",
        "mean_diff_rate",
        "median_diff_rate",
        "count"
    ]]
    .assign(
        mean_diff_rate=lambda x: (x["mean_diff_rate"] * 100).round(1),
        median_diff_rate=lambda x: (x["median_diff_rate"] * 100).round(1),
    )
    .rename(columns={
        "mean_diff_rate": "平均割高率(%)",
        "median_diff_rate": "中央値割高率(%)",
        "count": "件数",
    })
    .to_string(index=False)
)

print("\n===================================")
print(" 割安率ベスト地区（平均DIFF_RATE）")
print("===================================")
print(
    underpriced[[
        "DISTRICT_NAME",
        "mean_diff_rate",
        "median_diff_rate",
        "count"
    ]]
    .assign(
        mean_diff_rate=lambda x: (x["mean_diff_rate"] * 100).round(1),
        median_diff_rate=lambda x: (x["median_diff_rate"] * 100).round(1),
    )
    .rename(columns={
        "mean_diff_rate": "平均割安率(%)",
        "median_diff_rate": "中央値割安率(%)",
        "count": "件数",
    })
    .to_string(index=False)
)
