import os
import glob
import shutil
import json
import hashlib
import subprocess
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

def resolve_code_identity(production_mode: bool = False) -> str:
    """
    Resolves exact code version fingerprint using environment variables, git commit SHA,
    binary git diff checks, or source-tree file hashing.
    """
    env_sha = os.environ.get("TCELM_GIT_SHA")
    if env_sha and env_sha.strip():
        return env_sha.strip()

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        diff_bytes = subprocess.check_output(
            ["git", "diff", "HEAD", "--binary"], stderr=subprocess.DEVNULL
        )
        if diff_bytes.strip():
            diff_hash = hashlib.sha256(diff_bytes).hexdigest()[:8]
            return f"{git_sha}-dirty-{diff_hash}"
        return git_sha
    except Exception:
        if production_mode:
            raise RuntimeError("Code identity resolution failed in production mode: Git is not available and TCELM_GIT_SHA is unset.")

        # Non-production fallback: compute SHA-256 fingerprint over python source files
        source_dir = os.path.join(os.path.dirname(__file__), "..")
        py_files = sorted(glob.glob(os.path.join(source_dir, "**", "*.py"), recursive=True))
        hasher = hashlib.sha256()
        for pf in py_files:
            try:
                with open(pf, "rb") as f:
                    hasher.update(f.read())
            except OSError:
                pass
        return f"src_fingerprint_{hasher.hexdigest()[:12]}"

class BaseStage(ABC):
    """
    Abstract base class for disk-backed, sharded, restartable pipeline stages.
    """
    STAGE_SCHEMA_VERSION = "v0.1"

    def __init__(self, stage_name: str, output_dir: str, config: CorpusPipelineConfig, code_identity: Optional[str] = None):
        self.stage_name = stage_name
        self.config = config
        self.output_dir = output_dir
        self.code_identity = code_identity or resolve_code_identity(production_mode=getattr(config, "production_mode", False))
        self.stage_dir = os.path.join(output_dir, "stages", stage_name)
        os.makedirs(self.stage_dir, exist_ok=True)

        self.shard_io = ParquetShardIO(self.stage_dir)
        self.manifest = StageManifest(self.stage_dir)
        self.checkpoint = StageCheckpointManager(self.stage_dir)

    def get_additional_cache_inputs(self) -> Dict[str, str]:
        """
        Overridable hook for stages that depend on external artifacts (e.g. tokenizer.json).
        """
        return {}

    def is_completed(self) -> bool:
        man_data = self.manifest.load()
        if not man_data or man_data.get("status") != "SUCCESS":
            return False
        return man_data.get("config_hash") == self._compute_stage_cache_key()

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
            print(f"Stage `{self.stage_name}` already completed with matching cache key. Skipping.")
            return self.manifest.load() or {}

        if force or not self.is_completed():
            self.purge_stage_outputs()

        print(f"=== Executing Stage: {self.stage_name} ===")
        results = self.run_stage()
        
        self.manifest.save(
            stage_name=self.stage_name,
            status="SUCCESS",
            config_hash=self._compute_stage_cache_key(),
            code_commit=self.code_identity,
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

    def _compute_stage_cache_key(self) -> str:
        canonical_config_json = json.dumps(self.config.to_dict(), sort_keys=True)
        upstream_hash = ""
        prev_stage = self._get_upstream_stage_name()
        if prev_stage:
            prev_man_path = os.path.join(self.output_dir, "stages", prev_stage, "manifest.json")
            if os.path.exists(prev_man_path):
                upstream_hash = StageManifest.compute_file_hash(prev_man_path)

        add_inputs = self.get_additional_cache_inputs()
        add_inputs_json = json.dumps(add_inputs, sort_keys=True)

        payload = {
            "stage_name": self.stage_name,
            "stage_schema_version": self.STAGE_SCHEMA_VERSION,
            "config": canonical_config_json,
            "code_identity": self.code_identity,
            "upstream_manifest_sha256": upstream_hash,
            "additional_cache_inputs": add_inputs_json
        }

        key_json = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(key_json.encode("utf-8")).hexdigest()

    @abstractmethod
    def run_stage(self) -> Dict[str, Any]:
        pass
