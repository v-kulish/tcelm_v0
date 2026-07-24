import hashlib
from typing import Dict, Any
from collections import defaultdict
from tqdm import tqdm
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO

class Stage07Split(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("07_split", output_dir, config)
        self.input_io = ParquetShardIO(f"{output_dir}/stages/06_decontaminate")

    def run_stage(self) -> Dict[str, Any]:
        all_input = list(self.input_io.read_shards())
        if not all_input:
            raise RuntimeError("Stage '07_split' received 0 input records from Stage 06.")

        split_records = []
        record_counts = {}
        token_counts = {}

        all_splits = ["train", "validation", "test", "trajectory_holdout"]
        split_metrics = {
            s: {
                "split_groups": set(),
                "parents": set(),
                "segments": 0,
                "provisional_tokens": 0
            } for s in all_splits
        }

        val_share = getattr(self.config.splits, "val", 0.0010)
        test_share = getattr(self.config.splits, "test", 0.0010)
        holdout_share = getattr(self.config.splits, "trajectory_holdout", 0.0010)

        print(f"Stage 07 Split Assignment: Assigning splits across {len(all_input):,} decontaminated documents...")
        for rec in tqdm(all_input, desc="Assigning Splits", unit="doc"):
            doc_id = rec.get("document_id") or rec.get("doc_id")
            parent_id = rec.get("parent_document_id") or doc_id
            split_group_id = rec.get("split_group_id") or parent_id

            hash_str = f"{split_group_id}:{self.config.seed}"
            score = int(hashlib.md5(hash_str.encode('utf-8')).hexdigest()[:8], 16) / 0xFFFFFFFF

            if score < val_share:
                split = "validation"
            elif score < (val_share + test_share):
                split = "test"
            elif score < (val_share + test_share + holdout_share):
                split = "trajectory_holdout"
            else:
                split = "train"

            rec["split"] = split
            rec["document_id"] = doc_id
            rec["split_group_id"] = split_group_id
            split_records.append(rec)

            src = rec["source"]
            toks = len(rec["normalized_text"].split())
            record_counts[src] = record_counts.get(src, 0) + 1
            token_counts[src] = token_counts.get(src, 0) + toks

            split_metrics[split]["split_groups"].add(split_group_id)
            split_metrics[split]["parents"].add(parent_id)
            split_metrics[split]["segments"] += 1
            split_metrics[split]["provisional_tokens"] += toks

        written_shards = self.shard_io.write_records_to_shards(split_records, shard_prefix="part")

        # Format summary split dictionary for output
        split_summary = {}
        for s, m in split_metrics.items():
            split_summary[s] = {
                "split_group_count": len(m["split_groups"]),
                "parent_document_count": len(m["parents"]),
                "segment_count": m["segments"],
                "provisional_tokens": m["provisional_tokens"]
            }

        print(f"Stage 07 Split Assignment complete: Assigned {len(split_records):,} documents to splits: {split_summary}.")

        return {
            "record_counts": record_counts,
            "token_counts": token_counts,
            "split_metrics": split_summary,
            "output_hashes": {"shard_count": len(written_shards)}
        }
