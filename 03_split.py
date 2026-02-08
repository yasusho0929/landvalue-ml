import pandas as pd

# ===== 読み込み =====
df = pd.read_csv("test.csv", encoding="cp932")

# 念のため正規化（全角ブレ対策）
df["種類"] = df["種類"].astype(str).str.normalize("NFKC")

# ===== フィルタ =====
df_land_building = df[df["種類"] == "宅地(土地と建物)"]
df_land_only = df[df["種類"] == "宅地(土地)"]

# ===== 保存 =====
df_land_building.to_csv(
    "merged_land_and_building.csv",
    index=False,
    encoding="utf-8-sig"
)

df_land_only.to_csv(
    "merged_land_only.csv",
    index=False,
    encoding="utf-8-sig"
)

print("宅地(土地と建物):", df_land_building.shape)
print("宅地(土地):", df_land_only.shape)
