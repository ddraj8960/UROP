#!/usr/bin/env python3
"""
M3 Script: Build the scoped corpus and local indexes.
"""
import argparse
import sys
from pathlib import Path

from hmafact.schemas.evidence import BuildIndexRequest
from hmafact.retrieval.corpus import build_index

def main():
    parser = argparse.ArgumentParser(description="Build HMA-Fact local index.")
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--output-dir", default="data/indices", help="Output directory for index")
    args = parser.parse_args()

    print(f"Building index from {args.data_dir} into {args.output_dir}...")
    req = BuildIndexRequest(data_dir=args.data_dir, output_dir=args.output_dir)
    try:
        resp = build_index(req)
        print(f"Index built successfully in {resp.gpu_time_seconds:.2f}s.")
        print(f"Passages indexed: {resp.passage_count}")
        print(f"Manifest written to: {resp.manifest_path}")
    except Exception as e:
        print(f"Error building index: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
