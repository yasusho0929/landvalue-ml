#!/usr/bin/env python3
"""01_SplitData.py で作成した分割データをマニフェストから復元するスクリプト。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


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


def load_manifests(artifact_root: Path) -> list[Path]:
    chunks_root = artifact_root / "chunks"
    print(f"[SCAN] マニフェスト探索開始: {chunks_root}")
    if not chunks_root.exists():
        print(f"[SCAN] chunks ディレクトリが見つかりません: {chunks_root}")
        return []

    manifests = sorted(chunks_root.rglob("manifest.json"))
    print(f"[SCAN] マニフェスト探索完了: {len(manifests)} 件")
    for m in manifests:
        print(f"[SCAN] - {m}")
    return manifests


def restore_one(manifest_path: Path, restore_root: Path, overwrite: bool) -> None:
    print("=" * 80)
    print(f"[RESTORE] 復元開始: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rel_path = Path(manifest["original_relative_path"])
    target_path = restore_root / rel_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not overwrite:
        print(f"[RESTORE] スキップ: 既存ファイルあり (overwrite=False): {target_path}")
        return

    parts = manifest["parts"]
    expected_size = int(manifest["original_size"])
    expected_sha256 = manifest["original_sha256"]

    temp_path = target_path.with_suffix(target_path.suffix + ".restoring")
    written = 0

    with temp_path.open("wb") as dst:
        for idx, part_name in enumerate(parts, start=1):
            part_path = manifest_path.parent / part_name
            if not part_path.exists():
                raise FileNotFoundError(f"パートファイルが見つかりません: {part_path}")

            print(f"[RESTORE] パート結合: {idx}/{len(parts)} <- {part_path}")
            data = part_path.read_bytes()
            dst.write(data)
            written += len(data)
            print(f"[RESTORE] パート結合完了: {part_name} ({len(data):,} bytes)")

    print(f"[RESTORE] 書き込みサイズ: {written:,} bytes")
    if written != expected_size:
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            f"サイズ不一致: expected={expected_size:,} bytes, actual={written:,} bytes, target={target_path}"
        )

    actual_sha256 = sha256_of_file(temp_path)
    if actual_sha256 != expected_sha256:
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA256 不一致: expected={expected_sha256}, actual={actual_sha256}, target={target_path}"
        )

    temp_path.replace(target_path)
    print(f"[RESTORE] 復元完了: {target_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="01_SplitData.py で作成したマニフェストを使ってファイルを復元します。"
    )
    parser.add_argument(
        "--artifact-root",
        default="99_Spliit/_split_artifacts",
        help="01_SplitData.py の出力先ディレクトリ (default: 99_Spliit/_split_artifacts)",
    )
    parser.add_argument(
        "--restore-root",
        default=".",
        help="復元先ルート。manifest の相対パス配下に復元されます (default: .)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存ファイルがある場合に上書きする",
    )

    args = parser.parse_args()

    artifact_root = Path(args.artifact_root).resolve()
    restore_root = Path(args.restore_root).resolve()

    print("[INFO] 02_MixData.py 開始")
    print(f"[INFO] artifact_root={artifact_root}")
    print(f"[INFO] restore_root={restore_root}")
    print(f"[INFO] overwrite={args.overwrite}")

    manifests = load_manifests(artifact_root)
    if not manifests:
        print("[INFO] 復元対象のマニフェストがありません。処理終了。")
        return

    for manifest_path in manifests:
        restore_one(manifest_path=manifest_path, restore_root=restore_root, overwrite=args.overwrite)

    print("[INFO] 02_MixData.py 完了")


if __name__ == "__main__":
    main()
