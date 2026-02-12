#!/usr/bin/env python3
"""100MB を超えるファイルを 100MB ごとに分割し、復元用マニフェストを作成するスクリプト。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

DEFAULT_SPLIT_SIZE_MB = 100
MB = 1024 * 1024


def sha256_of_file(path: Path) -> str:
    print(f"[HASH] SHA256 計算開始: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    value = digest.hexdigest()
    print(f"[HASH] SHA256 計算完了: {path} -> {value}")
    return value


def find_large_files(root: Path, split_size_bytes: int, work_dir: Path) -> list[Path]:
    print(f"[SCAN] 走査開始: root={root}")
    large_files: list[Path] = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        # 作業ディレクトリ配下は対象外
        try:
            p.relative_to(work_dir)
            continue
        except ValueError:
            pass

        # .git 配下は対象外
        if ".git" in p.parts:
            continue

        size = p.stat().st_size
        if size > split_size_bytes:
            print(f"[SCAN] 対象ファイル発見: {p} ({size:,} bytes)")
            large_files.append(p)

    print(f"[SCAN] 走査完了: 対象 {len(large_files)} 件")
    return large_files


def split_file(file_path: Path, root: Path, output_root: Path, split_size_bytes: int) -> None:
    rel_path = file_path.relative_to(root)
    file_size = file_path.stat().st_size
    total_parts = math.ceil(file_size / split_size_bytes)

    chunk_dir = output_root / "chunks" / rel_path
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"[SPLIT] 分割開始: {file_path}")
    print(f"[SPLIT] 元ファイル相対パス: {rel_path}")
    print(f"[SPLIT] サイズ: {file_size:,} bytes / パート数: {total_parts}")

    with file_path.open("rb") as src:
        for idx in range(total_parts):
            part_no = idx + 1
            part_name = f"part_{part_no:05d}.bin"
            part_path = chunk_dir / part_name

            print(f"[SPLIT] パート書き込み開始: {part_no}/{total_parts} -> {part_path}")
            data = src.read(split_size_bytes)
            with part_path.open("wb") as part_file:
                part_file.write(data)
            print(f"[SPLIT] パート書き込み完了: {part_name} ({len(data):,} bytes)")

    original_sha256 = sha256_of_file(file_path)
    manifest = {
        "version": 1,
        "original_relative_path": str(rel_path),
        "original_size": file_size,
        "original_sha256": original_sha256,
        "split_size_bytes": split_size_bytes,
        "total_parts": total_parts,
        "parts": [f"part_{i + 1:05d}.bin" for i in range(total_parts)],
    }

    manifest_path = chunk_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SPLIT] マニフェスト作成完了: {manifest_path}")
    print(f"[SPLIT] 分割完了: {file_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ルート配下の 100MB 超ファイルを分割し、復元用マニフェストを作成します。"
    )
    parser.add_argument("--root", default=".", help="走査対象ルートディレクトリ (default: .)")
    parser.add_argument(
        "--output",
        default="99_Spliit/_split_artifacts",
        help="分割ファイルとマニフェストの出力先 (default: 99_Spliit/_split_artifacts)",
    )
    parser.add_argument(
        "--split-size-mb",
        type=int,
        default=DEFAULT_SPLIT_SIZE_MB,
        help="分割サイズ(MB)。この値より大きいファイルを対象に分割します (default: 100)",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_root = Path(args.output).resolve()
    split_size_bytes = args.split_size_mb * MB

    print("[INFO] 01_SplitData.py 開始")
    print(f"[INFO] root={root}")
    print(f"[INFO] output={output_root}")
    print(f"[INFO] split_size={split_size_bytes:,} bytes ({args.split_size_mb}MB)")

    output_root.mkdir(parents=True, exist_ok=True)

    targets = find_large_files(root=root, split_size_bytes=split_size_bytes, work_dir=output_root)

    if not targets:
        print("[INFO] 分割対象のファイルはありませんでした。処理終了。")
        return

    for target in targets:
        split_file(
            file_path=target,
            root=root,
            output_root=output_root,
            split_size_bytes=split_size_bytes,
        )

    print("[INFO] 01_SplitData.py 完了")


if __name__ == "__main__":
    main()
