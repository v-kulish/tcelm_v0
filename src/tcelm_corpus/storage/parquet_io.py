import os
import glob
from typing import List, Dict, Any, Iterator, Optional
import pyarrow as pa
import pyarrow.parquet as pq

class ParquetShardIO:
    """
    Handles streaming reads and writes of sharded Parquet files for pipeline stages.
    """
    def __init__(self, stage_dir: str, shard_size: int = 5000):
        self.stage_dir = stage_dir
        self.shard_size = shard_size
        os.makedirs(stage_dir, exist_ok=True)

    def write_records_to_shards(
        self,
        records: Iterator[Dict[str, Any]],
        schema: Optional[pa.Schema] = None,
        shard_prefix: str = "part"
    ) -> List[str]:
        os.makedirs(self.stage_dir, exist_ok=True)
        written_shards = []
        batch = []
        shard_idx = len(glob.glob(os.path.join(self.stage_dir, f"{shard_prefix}_*.parquet")))

        for rec in records:
            batch.append(rec)
            if len(batch) >= self.shard_size:
                shard_path = self._write_batch(batch, shard_prefix, shard_idx, schema)
                written_shards.append(shard_path)
                shard_idx += 1
                batch = []

        if batch:
            shard_path = self._write_batch(batch, shard_prefix, shard_idx, schema)
            written_shards.append(shard_path)

        return written_shards

    def _write_batch(
        self,
        batch: List[Dict[str, Any]],
        shard_prefix: str,
        shard_idx: int,
        schema: Optional[pa.Schema]
    ) -> str:
        os.makedirs(self.stage_dir, exist_ok=True)
        shard_filename = f"{shard_prefix}_{shard_idx:05d}.parquet"
        shard_path = os.path.join(self.stage_dir, shard_filename)
        
        table = pa.Table.from_pylist(batch, schema=schema)
        pq.write_table(table, shard_path, compression="SNAPPY")
        return shard_path

    def read_shards(self, shard_prefix: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        if shard_prefix is None or shard_prefix == "*":
            shard_files = sorted(glob.glob(os.path.join(self.stage_dir, "*.parquet")))
        else:
            shard_files = sorted(glob.glob(os.path.join(self.stage_dir, f"{shard_prefix}_*.parquet")))

        for shard_file in shard_files:
            table = pq.read_table(shard_file)
            for row in table.to_pylist():
                yield row

    def list_shards(self, shard_prefix: Optional[str] = None) -> List[str]:
        if shard_prefix is None or shard_prefix == "*":
            return sorted(glob.glob(os.path.join(self.stage_dir, "*.parquet")))
        return sorted(glob.glob(os.path.join(self.stage_dir, f"{shard_prefix}_*.parquet")))
