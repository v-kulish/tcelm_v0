import os
import glob
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..storage.parquet_io import ParquetShardIO
from ..storage.manifest import StageManifest
from ..storage.checkpoint import StageCheckpointManager
from ..config import CorpusPipelineConfig

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
        return man_data is not None and man_data.get("status") == "SUCCESS"

    def purge_stage_outputs(self):
        """
        Removes existing parquet shards, manifests, and checkpoints for transactional forced reruns.
        """
        for f in glob.glob(os.path.join(self.stage_dir, "*.parquet")):
            try:
                os.remove(f)
            except OSError:
                pass
        
        manifest_path = os.path.join(self.stage_dir, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                os.remove(manifest_path)
            except OSError:
                pass

        chk_path = os.path.join(self.stage_dir, "completed_shards.json")
        if os.path.exists(chk_path):
            try:
                os.remove(chk_path)
            except OSError:
                pass

    def execute(self, force: bool = False) -> Dict[str, Any]:
        if not force and self.is_completed():
            print(f"Stage `{self.stage_name}` already completed. Skipping.")
            return self.manifest.load() or {}

        if force:
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

    def _compute_config_hash(self) -> str:
        s = f"{self.config.corpus_version}:{self.config.seed}:{self.config.target_scale_tokens}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    @abstractmethod
    def run_stage(self) -> Dict[str, Any]:
        pass
