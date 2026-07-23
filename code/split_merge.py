"""
大文件拆分/合并工具 — 用于突破 GitHub 2GB 单文件限制

拆分：
    python split_merge.py split model/step_015000.pt
    python split_merge.py split data_stage1/pretrain_t2t_cleaned_two.jsonl --chunk 1900

合并：
    python split_merge.py merge model/step_015000.pt
"""

import argparse
import os
import sys


def split_file(filepath, chunk_mb=1900):
    """将大文件拆分为 .part00, .part01, ..."""
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        sys.exit(1)

    chunk_size = chunk_mb * 1024 * 1024
    total_size = os.path.getsize(filepath)
    expected_parts = (total_size + chunk_size - 1) // chunk_size

    print(f"拆分: {filepath} ({total_size/1024/1024:.0f} MB → {expected_parts} 块, 每块 ≤{chunk_mb}MB)")

    with open(filepath, "rb") as src:
        i = 0
        while True:
            data = src.read(chunk_size)
            if not data:
                break
            part_name = f"{filepath}.part{i:02d}"
            with open(part_name, "wb") as dst:
                dst.write(data)
            print(f"  → {part_name} ({len(data)/1024/1024:.0f} MB)")
            i += 1

    print(f"完成，共 {i} 块")


def merge_file(filepath):
    """将 .part00, .part01, ... 合并回原文件"""
    base_dir = os.path.dirname(filepath) or "."
    basename = os.path.basename(filepath)

    parts = sorted(
        [f for f in os.listdir(base_dir) if f.startswith(basename + ".part")],
        key=lambda x: x.rsplit(".part", 1)[1],
    )

    if not parts:
        print(f"找不到分块文件: {filepath}.part*")
        sys.exit(1)

    print(f"合并: {len(parts)} 个分块 → {filepath}")
    total = 0
    with open(filepath, "wb") as dst:
        for p in parts:
            part_path = os.path.join(base_dir, p)
            with open(part_path, "rb") as src:
                data = src.read()
                dst.write(data)
                total += len(data)
            print(f"  ← {p} ({len(data)/1024/1024:.0f} MB)")

    print(f"完成，合并后大小: {total/1024/1024:.0f} MB")


def main():
    parser = argparse.ArgumentParser(description="大文件拆分/合并工具")
    parser.add_argument("mode", choices=["split", "merge"], help="split=拆分 / merge=合并")
    parser.add_argument("filepath", help="目标文件路径")
    parser.add_argument("--chunk", type=int, default=1900, help="每块大小 (MB)，默认 1900")

    args = parser.parse_args()

    if args.mode == "split":
        split_file(args.filepath, args.chunk)
    else:
        merge_file(args.filepath)


if __name__ == "__main__":
    main()
