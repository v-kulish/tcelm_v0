import os
import json
import hashlib
from typing import Dict, Any
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO
from ..storage.manifest import StageManifest

class Stage10Freeze(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("10_freeze", output_dir, config)
        self.output_dir = output_dir

    def run_stage(self) -> Dict[str, Any]:
        layer_a_dir = os.path.join(self.output_dir, "stages", "07_split")
        layer_b_dir = os.path.join(self.output_dir, "stages", "09_tokenize_select")

        checksums = {
            "layer_a_shards": {},
            "layer_b_shards": {}
        }

        io_a = ParquetShardIO(layer_a_dir)
        for s in io_a.list_shards("split"):
            checksums["layer_a_shards"][os.path.basename(s)] = StageManifest.compute_file_hash(s)

        io_b = ParquetShardIO(layer_b_dir)
        for s in io_b.list_shards("layer_b"):
            checksums["layer_b_shards"][os.path.basename(s)] = StageManifest.compute_file_hash(s)

        freeze_file = os.path.join(self.stage_dir, "freeze_manifest.json")
        with open(freeze_file, "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2)

        print(f"Frozen corpus checksum manifest written to `{freeze_file}`.")
        return {
            "record_counts": {
                "layer_a_shards": len(checksums["layer_a_shards"]),
                "layer_b_shards": len(checksums["layer_b_shards"])
            },
            "output_hashes": {"freeze_manifest": freeze_file}
        }
