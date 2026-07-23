#!/usr/bin/env python3
import argparse
import sys
import os
from src.tcelm_corpus.runner import CorpusPipelineRunner

def main():
    parser = argparse.ArgumentParser(description="TCELM-Corpus-v0 Data Processing Pipeline")
    parser.add_argument("--config", type=str, default="config/corpus_v0_3b.json", help="Path to pipeline config JSON")
    parser.add_argument("--scale", type=str, default="50M", help="Target corpus scale (e.g. 50M, 1B, 3B, 5B)")
    parser.add_argument("--output-dir", type=str, default="output_corpus_run", help="Output directory for artifacts and reports")
    parser.add_argument("--max-records-per-source", type=int, default=None, help="Optional max records per source (for quick local debugging)")
    parser.add_argument("--force-restart", action="store_true", help="Force re-execution of completed stages")

    args = parser.parse_args()

    scale_map = {
        "50M": 50_000_000,
        "1B": 1_000_000_000,
        "3B": 3_000_000_000,
        "5B": 5_000_000_000
    }

    target_tokens = scale_map.get(args.scale.upper())
    if target_tokens is None:
        try:
            target_tokens = int(args.scale)
        except ValueError:
            print(f"Error: Invalid scale argument '{args.scale}'. Choose 50M, 1B, 3B, 5B or an integer number of tokens.")
            sys.exit(1)

    runner = CorpusPipelineRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        target_scale_tokens=target_tokens,
        max_records_per_source=args.max_records_per_source
    )

    summary = runner.run(force_restart=args.force_restart)
    print("\n--- Pipeline Run Summary ---")
    reports_dir = os.path.join(args.output_dir, "reports")
    if os.path.exists(reports_dir):
        print(f"Reports directory: {reports_dir}")
        for rep in sorted(os.listdir(reports_dir)):
            print(f"  - {rep}")

if __name__ == "__main__":
    main()
