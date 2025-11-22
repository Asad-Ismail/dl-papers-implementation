#!/usr/bin/env python3
"""
Synthetic hallucination data generator.

This script reads an annotations JSON file (with either a top-level "data" list or a bare list of entries),
and for each entry, produces a synthetic hallucinated ground-truth by randomly sampling a different label
from the dataset's labels, writing out a new JSON file with modified entries.

Usage:
  python scripts/synthesize_hallucination.py \
      --input_json data/annotations/chair.json \
      [--output_json data/annotations/chair_synth.json] \
      [--seed 123]
"""
import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic hallucinated annotations by random label replacement"
    )
    parser.add_argument(
        '--input_json', type=str, required=True,
        help="Path to original annotations JSON (dict with 'data' or list)"
    )
    parser.add_argument(
        '--output_json', type=str, default=None,
        help="Path to save synthetic annotations JSON"
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load data
    with open(args.input_json, 'r') as f:
        raw = json.load(f)

    if isinstance(raw, dict) and 'data' in raw:
        entries = raw['data']
        wrapper = 'data'
    elif isinstance(raw, list):
        entries = raw
        wrapper = None
    else:
        raise ValueError(
            "Unsupported JSON format: expected dict with 'data' key or a list of entries"
        )

    # Collect all possible labels from the dataset
    labels = []
    for entry in entries:
        # Determine GT key
        if 'ground_truth' in entry:
            gt_val = entry['ground_truth']
        elif 'answer' in entry:
            gt_val = entry['answer']
        else:
            continue
        if isinstance(gt_val, list):
            labels.extend(gt_val)
        else:
            labels.append(gt_val)
    labels = list(set(labels))
    if not labels:
        raise ValueError("No labels found in the dataset to sample for hallucination.")

    # Generate synthetic entries
    synth_entries = []
    for entry in entries:
        # Determine GT key and original value
        if 'ground_truth' in entry:
            gt_key = 'ground_truth'
        elif 'answer' in entry:
            gt_key = 'answer'
        else:
            # skip entries without a recognized label field
            synth_entries.append(entry)
            continue
        original = entry[gt_key]
        # Choose a random label not equal to the original
        candidates = [lbl for lbl in labels if lbl != original]
        if candidates:
            synth_val = random.choice(candidates)
        else:
            # fallback if no other labels
            synth_val = "unknown"
        # Copy and override
        new_entry = dict(entry)
        new_entry[gt_key] = synth_val
        # Store the original value
        new_entry[f"original_{gt_key}"] = original
        synth_entries.append(new_entry)

    # Prepare output structure
    if wrapper == 'data':
        out_data = {'data': synth_entries}
    else:
        out_data = synth_entries

    # Determine output path
    if args.output_json:
        output_path = args.output_json
    else:
        base, ext = os.path.splitext(args.input_json)
        output_path = f"{base}_synth{ext}"

    # Ensure directory exists
    out_dir = os.path.dirname(output_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(out_data, f, indent=2)

    print(f"Synthesized {len(synth_entries)} entries -> {output_path}")


if __name__ == '__main__':
    main()
