import zipfile
import glob
import os

folder = "path/to/folder"

# ** 再帰（サブフォルダ含む）検索 **
zip_files = glob.glob(os.path.join(folder, "**", "*.zip"), recursive=True)

for z in zip_files:
    with zipfile.ZipFile(z) as zip_ref:
        extract_dir = os.path.splitext(z)[0]  # ZIP と同名フォルダに展開
        os.makedirs(extract_dir, exist_ok=True)
        zip_ref.extractall(extract_dir)

