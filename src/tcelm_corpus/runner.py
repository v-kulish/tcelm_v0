import os
from typing import Optional, Dict, Any
from .config import CorpusPipelineConfig
from .stages import (
    Stage01Ingest,
    Stage02SelectPool,
    Stage03NormalizeClean,
    Stage04Segment,
    Stage05Dedup,
    Stage06Decontaminate,
    Stage07Split,
    Stage08TrainTokenizer,
    Stage09TokenizeSelect,
    Stage10Freeze,
    Stage11GenerateViews,
    Stage12StatsReports
)

class CorpusPipelineRunner:
    """
    Disk-backed, sharded, 12-stage production orchestrator for TCELM-Corpus-v0.
    """
    def __init__(self, config_path: str, output_dir: str, target_scale_tokens: Optional[int] = None, max_records_per_source: Optional[int] = None):
        self.config = CorpusPipelineConfig.load_from_json(config_path, target_scale_tokens)
        if max_records_per_source is not None:
            self.config.max_records_per_source = max_records_per_source
        self.output_dir = output_dir

        self.stages = [
            Stage01Ingest(output_dir, self.config),
            Stage02SelectPool(output_dir, self.config),
            Stage03NormalizeClean(output_dir, self.config),
            Stage04Segment(output_dir, self.config),
            Stage05Dedup(output_dir, self.config),
            Stage06Decontaminate(output_dir, self.config),
            Stage07Split(output_dir, self.config),
            Stage08TrainTokenizer(output_dir, self.config),
            Stage09TokenizeSelect(output_dir, self.config),
            Stage10Freeze(output_dir, self.config),
            Stage11GenerateViews(output_dir, self.config),
            Stage12StatsReports(output_dir, self.config)
        ]

    def run(self, force_restart: bool = False) -> Dict[str, Any]:
        print(f"=== Starting TCELM Corpus Pipeline (Target Scale: {self.config.target_scale_tokens:,} tokens) ===")
        
        summary = {}
        for stage in self.stages:
            stage_res = stage.execute(force=force_restart)
            summary[stage.stage_name] = stage_res

        print("=== TCELM Corpus Pipeline Completed Successfully ===")
        return summary
