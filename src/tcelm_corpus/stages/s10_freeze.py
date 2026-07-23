import os
import json
from typing import Dict, Any
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO
from ..storage.manifest import StageManifest

class Stage10Freeze(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("10_freeze", output_dir, config)
        self.output_dir = output_dir

    def run_stage(self) -> Dict[str, Any]:
        stage_09_dir = os.path.join(self.output_dir, "stages", "09_tokenize_select")
        layer_a_dir = os.path.join(stage_09_dir, "layer_a_selected")
        layer_b_dir = os.path.join(stage_09_dir, "layer_b_selected")

        io_a = ParquetShardIO(layer_a_dir)
        io_b = ParquetShardIO(layer_b_dir)

        docs_a = [r.get("document_id") or r.get("doc_id") for r in io_a.read_shards()]
        docs_b = [r.get("document_id") or r.get("doc_id") for r in io_b.read_shards()]

        if not docs_a or not docs_b:
            raise RuntimeError("Stage '10_freeze' received 0 selected records from Stage 09.")

        if docs_a != docs_b:
            raise RuntimeError(f"Layer A and Layer B document ID mismatch in Stage 10 Freeze: Layer A has {len(docs_a)} docs, Layer B has {len(docs_b)} docs.")

        checksums = {
            "layer_a_shards": {},
            "layer_b_shards": {}
        }

        for s in io_a.list_shards():
            checksums["layer_a_shards"][os.path.basename(s)] = StageManifest.compute_file_hash(s)

        for s in io_b.list_shards():
            checksums["layer_b_shards"][os.path.basename(s)] = StageManifest.compute_file_hash(s)

        freeze_file = os.path.join(self.stage_dir, "freeze_manifest.json")
        with open(freeze_file, "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2)

        print(f"Frozen corpus checksum manifest written to `{freeze_file}` ({len(docs_a):,} documents verified 1-to-1).")
        return {
            "record_counts": {
                "verified_documents": len(docs_a),
                "layer_a_shards": len(checksums["layer_a_shards"]),
                "layer_b_shards": len(checksums["layer_b_shards"])
            },
            "output_hashes": {"freeze_manifest": freeze_file}
        }
