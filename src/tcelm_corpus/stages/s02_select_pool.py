from typing import Dict, Any, List
from collections import defaultdict
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO

class Stage02SelectPool(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("02_select_pool", output_dir, config)
        self.input_io = ParquetShardIO(f"{output_dir}/stages/01_ingest")

    def run_stage(self) -> Dict[str, Any]:
        source_records = defaultdict(list)
        
        # Read all ingested candidates from stage 01
        for row in self.input_io.read_shards():
            source_records[row["source"]].append(row)

        retained_records = []
        record_counts = {}
        token_counts = {}

        for source_cfg in self.config.sources:
            source_name = source_cfg.name
            target_quota = self.config.get_source_quota(source_name)
            retained_target_tokens = int(target_quota * self.config.oversampling_multiplier)

            records = source_records.get(source_name, [])
            # Sort by deterministic priority (q(d)) in ascending order
            records.sort(key=lambda x: x["priority"])

            accumulated_tokens = 0
            accumulated_docs = 0

            for rec in records:
                retained_records.append(rec)
                accumulated_tokens += rec.get("provisional_tokens", 0)
                accumulated_docs += 1

                if accumulated_tokens >= retained_target_tokens:
                    break

            record_counts[source_name] = accumulated_docs
            token_counts[source_name] = accumulated_tokens
            print(f"Source `{source_name}` retained pool: {accumulated_docs:,} docs, {accumulated_tokens:,} tokens (Target: {retained_target_tokens:,}).")

        written_shards = self.shard_io.write_records_to_shards(retained_records, shard_prefix="pool")

        return {
            "record_counts": record_counts,
            "token_counts": token_counts,
            "output_hashes": {"shard_count": len(written_shards)}
        }
