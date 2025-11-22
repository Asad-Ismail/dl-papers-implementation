#!/usr/bin/env python3
"""
Script to preprocess and split annotation datasets into train/val/test splits.
"""
import os
import sys
import json
import argparse
import random
from src.config import Config


def split_list(lst, val_frac, test_frac):
    """
    Split list into train, val, test based on given fractions.
    """
    n = len(lst)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    # Ensure at least 1 sample if fractions > 0 and list is non-empty
    if n_val == 0 and val_frac > 0 and n > 0:
        n_val = 1
    if n_test == 0 and test_frac > 0 and n > 0:
        n_test = 1
    n_train = n - n_val - n_test
    if n_train < 0:
        # Adjust: allocate leftover to train
        n_train = max(n - n_val - n_test, 0)
    train = lst[:n_train]
    val = lst[n_train:n_train + n_val]
    test = lst[n_train + n_val:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Preprocess and split annotation JSON files.")
    parser.add_argument(
        '--config', type=str, default='configs/default.yaml',
        help='Path to the default configuration YAML file.'
    )
    parser.add_argument(
        '--annotations_dir', type=str, default=None,
        help='Path to the annotations directory (overrides config).'
    )
    parser.add_argument(
        '--val_frac', type=float, default=0.1,
        help='Fraction of data to use for validation set.'
    )
    parser.add_argument(
        '--test_frac', type=float, default=0.1,
        help='Fraction of data to use for test set.'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for shuffling.'
    )
    args = parser.parse_args()

    # Load configuration
    cfg = Config(default_path=args.config)
    ann_dir = args.annotations_dir or cfg['data']['annotations_dir']
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"Annotations directory not found: {ann_dir}")

    # Prepare random seed
    random.seed(args.seed)

    # Process each JSON file in annotations_dir
    for fname in os.listdir(ann_dir):
        if not fname.endswith('.json'):
            continue
        if any(s in fname for s in ['_train.json', '_val.json', '_test.json']):
            continue
        path = os.path.join(ann_dir, fname)
        with open(path, 'r') as f:
            data = json.load(f)
        # Expecting a top-level 'data' key or a list
        if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
            records = data['data']
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError(f"Unexpected JSON structure in {path}")

        # Shuffle records
        random.shuffle(records)
        train, val, test = split_list(records, args.val_frac, args.test_frac)

        basename = os.path.splitext(fname)[0]
        out_train = os.path.join(ann_dir, f"{basename}_train.json")
        out_val = os.path.join(ann_dir, f"{basename}_val.json")
        out_test = os.path.join(ann_dir, f"{basename}_test.json")

        # Write out splits
        for subset, out_path in zip(
            [train, val, test], [out_train, out_val, out_test]
        ):
            content = {'data': subset} if isinstance(data, dict) else subset
            with open(out_path, 'w') as wf:
                json.dump(content, wf, indent=2)
            print(f"Wrote {len(subset)} records to {out_path}")

    print("Preprocessing completed.")


if __name__ == '__main__':
    main()
