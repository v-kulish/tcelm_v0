import os
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

class StageManifest:
    """
    Records metadata, input/output artifact SHA-256 hashes, record/token counters,
    code commit, and completion timestamps for pipeline stages.
    """
    def __init__(self, stage_dir: str):
        self.stage_dir = stage_dir
        self.manifest_path = os.path.join(stage_dir, "manifest.json")

    def load(self) -> Optional[Dict[str, Any]]:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def save(
        self,
        stage_name: str,
        status: str,
        code_commit: str = "main",
        input_hashes: Optional[Dict[str, str]] = None,
        output_hashes: Optional[Dict[str, str]] = None,
        source_revisions: Optional[Dict[str, str]] = None,
        record_counts: Optional[Dict[str, int]] = None,
        token_counts: Optional[Dict[str, int]] = None,
        rejection_counts: Optional[Dict[str, int]] = None,
        config_hash: str = "",
        seed: int = 42
    ) -> Dict[str, Any]:
        os.makedirs(self.stage_dir, exist_ok=True)
        data = {
            "stage_name": stage_name,
            "status": status,
            "code_commit": code_commit,
            "config_hash": config_hash,
            "seed": seed,
            "input_hashes": input_hashes or {},
            "output_hashes": output_hashes or {},
            "source_revisions": source_revisions or {},
            "record_counts": record_counts or {},
            "token_counts": token_counts or {},
            "rejection_counts": rejection_counts or {},
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

    @staticmethod
    def compute_file_hash(filepath: str) -> str:
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()
