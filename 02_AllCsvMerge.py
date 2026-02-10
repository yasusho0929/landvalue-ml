"""このスクリプトは、指定フォルダ（必要に応じてout）内のCSVを結合し、マージCSVを出力します。"""

import argparse
import glob
import os
import sys
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="指定フォルダ内のCSVをすべて縦結合し、<フォルダ名>_merge.csv を出力します。"
    )
    parser.add_argument(
        "--folder", "-f", required=True, help="CSVが入っているフォルダへのパス"
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="サブフォルダも含めてCSVを探す場合に指定"
    )
    parser.add_argument(
        "--encoding", "-e", default="utf-8",
        help="入力CSVの文字エンコーディング（既定: utf-8。日本語Windows系は cp932 推奨）"
    )
    parser.add_argument(
        "--output-encoding", "-O", default="utf-8-sig",
        help="出力CSVの文字エンコーディング（既定: utf-8-sig = Excel で文字化けしにくい）"
    )
    parser.add_argument(
        "--pattern", "-p", default="*.csv",
        help="検索パターン（既定: *.csv）"
    )
    parser.add_argument(
        "--sort", choices=["name", "none"], default="name",
        help="結合順序。name=ファイル名昇順 / none=並び替えなし（発見順）"
    )
    args = parser.parse_args()

    folder_path = os.path.abspath(args.folder)
    if not os.path.isdir(folder_path):
        print(f"[ERROR] フォルダが見つかりません: {folder_path}", file=sys.stderr)
        sys.exit(1)

    folder_name = os.path.basename(folder_path.rstrip(os.sep))
    output_path = os.path.join(folder_path, f"{folder_name}_merge.csv")

    # 検索パターンを構築
    if args.recursive:
        pattern = os.path.join(folder_path, "**", args.pattern)
        csv_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(folder_path, args.pattern)
        csv_files = glob.glob(pattern)

    # 自分自身の出力ファイル（過去の実行結果）が混ざらないように除外
    csv_files = [p for p in csv_files if os.path.abspath(p) != os.path.abspath(output_path)]

    if len(csv_files) == 0:
        print(f"[WARN] CSVが見つかりませんでした: {pattern}")
        sys.exit(0)

    if args.sort == "name":
        csv_files = sorted(csv_files, key=lambda x: os.path.basename(x).lower())

    print(f"対象CSV数: {len(csv_files)}")
    for i, p in enumerate(csv_files, 1):
        print(f"  {i:>3}: {os.path.relpath(p, start=folder_path)}")

    df_list = []
    columns_ref = None

    for path in csv_files:
        try:
            df = pd.read_csv(path, encoding=args.encoding)
        except UnicodeDecodeError:
            print(f"[ERROR] 文字コードエラー: {path} / encoding={args.encoding}", file=sys.stderr)
            print("        --encoding に cp932 や shift_jis を試してみてください。", file=sys.stderr)
            sys.exit(1)

        # 最初のCSVの列を基準に、完全一致を念のため検証（ユーザー前提が完全一致）
        if columns_ref is None:
            columns_ref = list(df.columns)
        else:
            if list(df.columns) != columns_ref:
                print(f"[ERROR] 列構成が一致しません: {path}", file=sys.stderr)
                print(f"        期待: {columns_ref}", file=sys.stderr)
                print(f"        実際: {list(df.columns)}", file=sys.stderr)
                sys.exit(1)

        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)

    # 出力
    merged_df.to_csv(output_path, index=False, encoding=args.output_encoding)
    print(f"\n✅ 出力完了: {output_path}")
    print(f"   行数: {len(merged_df)}  列数: {merged_df.shape[1]}  エンコーディング: {args.output_encoding}")

if __name__ == "__main__":
    main()