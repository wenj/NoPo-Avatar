#!/bin/bash

ROOT_DIR="datasets/HuGe100K/all"

for d in "$ROOT_DIR"/*; do
    if [ -d "$d" ]; then
        echo "Processing directory: $d"
        python datasets/HuGe100K/scripts/write_images.py --root_dir "$d"
        python datasets/HuGe100K/scripts/segment.py --root_dir "$d"
        python -m src.scripts.convert_huge100k --subset "$(basename "$d")"
    fi
done
