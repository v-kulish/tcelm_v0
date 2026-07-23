import os
import glob
import shutil
import json
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..storage.parquet_io import ParquetShardIO
from ..storage.manifest import StageManifest
from ..storage.checkpoint import StageCheckpointManager
from ..config import CorpusPipelineConfig

STAGE_ORDER = [
    "01_ingest", "02_select_pool", "03_normalize_clean", "04_segment",
    "05_dedup", "06_decontaminate", "07_split", "08_train_tokenizer",
    "09_tokenize_select", "10_freeze", "11_generate_views", "12_stats_reports"
]

class BaseStage(ABC):
    """
    Abstract base class for disk-backed, sharded, restartable pipeline stages.
    """
    def __init__(self, stage_name: str, output_dir: str, config: CorpusPipelineConfig):
        self.stage_name = stage_name
        self.config = config
        self.output_dir = output_dir
        self.stage_dir = os.path.join(output_dir, "stages", stage_name)
        os.makedirs(self.stage_dir, exist_ok=True)

        self.shard_io = ParquetShardIO(self.stage_dir)
        self.manifest = StageManifest(self.stage_dir)
        self.checkpoint = StageCheckpointManager(self.stage_dir)

    def is_completed(self) -> bool:
        man_data = self.manifest.load()
        if not man_data or man_data.get("status") != "SUCCESS":
            return False
        return man_data.get("config_hash") == self._compute_config_hash()

    def purge_stage_outputs(self):
        """
        Completely removes self.stage_dir including nested output subdirectories (e.g. layer_a_selected,
        layer_b_selected) for clean transactional forced reruns.
        """
        if os.path.exists(self.stage_dir):
            try:
                shutil.rmtree(self.stage_dir)
            except OSError:
                pass

        os.makedirs(self.stage_dir, exist_ok=True)

        self.shard_io = ParquetShardIO(self.stage_dir)
        self.manifest = StageManifest(self.stage_dir)
        self.checkpoint = StageCheckpointManager(self.stage_dir)

    def execute(self, force: bool = False) -> Dict[str, Any]:
        if not force and self.is_completed():
            print(f"Stage `{self.stage_name}` already completed with matching config and upstream hash. Skipping.")
            return self.manifest.load() or {}

        if force or not self.is_completed():
            self.purge_stage_outputs()

        print(f"=== Executing Stage: {self.stage_name} ===")
        results = self.run_stage()
        
        self.manifest.save(
            stage_name=self.stage_name,
            status="SUCCESS",
            config_hash=self._compute_config_hash(),
            seed=self.config.seed,
            record_counts=results.get("record_counts", {}),
            token_counts=results.get("token_counts", {}),
            rejection_counts=results.get("rejection_counts", {}),
            output_hashes=results.get("output_hashes", {})
        )
        return results

    def _get_upstream_stage_name(self) -> Optional[str]:
        if self.stage_name in STAGE_ORDER:
            idx = STAGE_ORDER.index(self.stage_name)
            if idx > 0:
                return STAGE_ORDER[idx - 1]
        return None

    def _compute_config_hash(self) -> str:
        canonical_json = json.dumps(self.config.to_dict(), sort_keys=True)
        upstream_hash = ""
        prev_stage = self._get_upstream_stage_name()
        if prev_stage:
            prev_man_path = os.path.join(self.output_dir, "stages", prev_stage, "manifest.json")
            if os.path.exists(prev_man_path):
                upstream_hash = StageManifest.compute_file_hash(prev_man_path)

        combo_key = f"{canonical_json}:{upstream_hash}"
        return hashlib.sha256(combo_key.encode("utf-8")).hexdigest()

    @abstractmethod
    def run_stage(self) -> Dict[str, Any]:
        pass
