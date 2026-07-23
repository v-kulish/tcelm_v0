import hashlib
from typing import Dict, Any
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
        split_counts = {}

        val_share = getattr(self.config.splits, "val", 0.0010)
        test_share = getattr(self.config.splits, "test", 0.0010)
        holdout_share = getattr(self.config.splits, "trajectory_holdout", 0.0010)

        for rec in all_input:
            doc_id = rec.get("document_id") or rec.get("doc_id")
            cluster_id = rec.get("dedup_cluster_id") or rec.get("parent_document_id") or doc_id
            hash_str = f"{cluster_id}:{self.config.seed}"
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
            split_records.append(rec)

            src = rec["source"]
            toks = len(rec["normalized_text"].split())
            record_counts[src] = record_counts.get(src, 0) + 1
            token_counts[src] = token_counts.get(src, 0) + toks
            split_counts[split] = split_counts.get(split, 0) + toks

        written_shards = self.shard_io.write_records_to_shards(split_records, shard_prefix="part")

        return {
            "record_counts": record_counts,
            "token_counts": token_counts,
            "rejection_counts": split_counts,
            "output_hashes": {"shard_count": len(written_shards)}
        }
