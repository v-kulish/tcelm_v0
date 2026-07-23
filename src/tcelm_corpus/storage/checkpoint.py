import os
import json
from typing import Set, Dict, Any, Optional

class StageCheckpointManager:
    """
    Manages shard-level completion status for crash-resume capability.
    """
    def __init__(self, stage_dir: str):
        self.checkpoint_file = os.path.join(stage_dir, "completed_shards.json")

    def get_completed_shards(self) -> Set[str]:
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                return set(json.load(f))
        return set()

    def mark_shard_completed(self, shard_filename: str):
        completed = self.get_completed_shards()
        completed.add(shard_filename)
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(sorted(list(completed)), f, indent=2)

    def is_shard_completed(self, shard_filename: str) -> bool:
        return shard_filename in self.get_completed_shards()
