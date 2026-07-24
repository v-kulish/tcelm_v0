#!/usr/bin/env python3
import argparse
import sys
import os
from src.tcelm_corpus.runner import CorpusPipelineRunner

def parse_token_count(val: str) -> int:
    val_upper = val.upper()
    multiplier = 1
    if val_upper.endswith("K"):
        multiplier = 1_000
        val_upper = val_upper[:-1]
    elif val_upper.endswith("M"):
        multiplier = 1_000_000
        val_upper = val_upper[:-1]
    elif val_upper.endswith("B"):
        multiplier = 1_000_000_000
        val_upper = val_upper[:-1]
    return int(float(val_upper) * multiplier)

def main():
    parser = argparse.ArgumentParser(description="TCELM-Corpus-v0 Data Processing Pipeline")
    parser.add_argument("--config", type=str, default="config/corpus_v0_3b.json", help="Path to pipeline config JSON")
    parser.add_argument("--scale", type=str, default="50M", help="Target corpus scale (e.g. 50M, 1B, 3B, 5B)")
    parser.add_argument("--smoke-total-tokens", type=str, default=None, help="Optional total proportional smoke token budget (e.g. 5M, 10M)")
    parser.add_argument("--output-dir", type=str, default="output_corpus_run", help="Output directory for artifacts and reports")
    parser.add_argument("--max-records-per-source", type=int, default=None, help="Optional max records per source (for quick local debugging)")
    parser.add_argument("--max-records-scanned-per-source", type=int, default=None, help="Optional hard ceiling on upstream records scanned per source")
    parser.add_argument("--max-consecutive-oversized-skips", type=int, default=None, help="Optional limit on consecutive oversized skips before terminating source scan")
    parser.add_argument("--force-restart", action="store_true", help="Force re-execution of completed stages")

    args = parser.parse_args()

    try:
        target_tokens = parse_token_count(args.scale)
    except ValueError:
        print(f"Error: Invalid scale argument '{args.scale}'. Choose 50M, 1B, 3B, 5B or an integer number of tokens.")
        sys.exit(1)

    smoke_tokens = None
    if args.smoke_total_tokens is not None:
        try:
            smoke_tokens = parse_token_count(args.smoke_total_tokens)
        except ValueError:
            print(f"Error: Invalid --smoke-total-tokens argument '{args.smoke_total_tokens}'.")
            sys.exit(1)

    runner = CorpusPipelineRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        target_scale_tokens=target_tokens,
        max_records_per_source=args.max_records_per_source,
        smoke_total_tokens=smoke_tokens,
        max_records_scanned_per_source=args.max_records_scanned_per_source,
        max_consecutive_oversized_skips_per_source=args.max_consecutive_oversized_skips
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
