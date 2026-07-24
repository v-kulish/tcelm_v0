import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

@dataclass
class SourceConfig:
    name: str
    category: str
    share: float
    quota_3b_tokens: int
    revision: str = "main"
    min_english_prob: float = 0.50
    min_doc_length: int = 128
    max_doc_length: int = 32768
    max_domain_share_of_quota: Optional[float] = None
    max_domain_share_of_corpus: Optional[float] = None
    subproject_shares: Optional[Dict[str, float]] = None
    field_shares: Optional[Dict[str, float]] = None
    max_answers: Optional[int] = None
    max_comments: Optional[int] = None
    max_site_share_of_quota: Optional[float] = None
    min_unigram_log_likelihood: Optional[float] = None
    max_ufffd_ratio: Optional[float] = 0.001
    max_code_log_token_ratio: Optional[float] = None

    @property
    def target_ratio(self) -> float:
        return self.share

@dataclass
class DedupConfig:
    num_bands: int = 16
    rows_per_band: int = 8
    minhash_num_perm: int = 128
    ngram_size: int = 20
    candidate_jaccard: float = 0.85
    final_jaccard: float = 0.90
    exact_paragraph_min_tokens: int = 50
    exact_paragraph_max_occurrences: int = 10

@dataclass
class DecontamConfig:
    shingle_size: int = 13
    min_token_overlap_ratio: float = 0.70

@dataclass
class SplitsConfig:
    train: float = 0.9970
    val: float = 0.0010
    test: float = 0.0010
    trajectory_holdout: float = 0.0010
    holdout_composition: Dict[str, float] = field(default_factory=dict)

@dataclass
class TokenizerConfig:
    vocab_size: int = 32768
    max_token_length_bytes: int = 32
    special_tokens: List[str] = field(default_factory=list)

@dataclass
class CorpusPipelineConfig:
    corpus_version: str = "TCELM-Corpus-v0"
    seed: int = 42
    oversampling_multiplier: float = 1.35
    target_scale_tokens: int = 3000000000
    production_mode: bool = False
    max_records_per_source: Optional[int] = None
    max_records_scanned_per_source: Optional[int] = None
    max_consecutive_oversized_skips_per_source: Optional[int] = None
    smoke_total_tokens: Optional[int] = None
    sources: List[SourceConfig] = field(default_factory=list)
    deduplication: DedupConfig = field(default_factory=DedupConfig)
    decontamination: DecontamConfig = field(default_factory=DecontamConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_source_quota(self, source_name: str) -> int:
        for s in self.sources:
            if s.name == source_name:
                return int(self.target_scale_tokens * s.share)
        raise ValueError(f"Unknown source: {source_name}")

    def get_source_initial_retained_pool(self, source_name: str) -> int:
        return int(self.get_source_quota(source_name) * self.oversampling_multiplier)

    @classmethod
    def load_from_json(cls, config_path: str, target_scale_tokens: Optional[int] = None) -> "CorpusPipelineConfig":
        path = Path(config_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        sources = [SourceConfig(**s) for s in data.get("sources", [])]
        dedup = DedupConfig(**data.get("deduplication", {}))
        decontam = DecontamConfig(**data.get("decontamination", {}))
        splits = SplitsConfig(**data.get("splits", {}))
        tok = TokenizerConfig(**data.get("tokenizer", {}))

        target_scale = target_scale_tokens if target_scale_tokens is not None else data.get("default_target_scale_tokens", 3000000000)
        is_prod = target_scale >= 1_000_000_000

        return cls(
            corpus_version=data.get("corpus_version", "TCELM-Corpus-v0"),
            seed=data.get("seed", 42),
            oversampling_multiplier=data.get("oversampling_multiplier", 1.35),
            target_scale_tokens=target_scale,
            production_mode=is_prod,
            max_records_per_source=data.get("max_records_per_source"),
            max_records_scanned_per_source=data.get("max_records_scanned_per_source"),
            max_consecutive_oversized_skips_per_source=data.get("max_consecutive_oversized_skips_per_source"),
            sources=sources,
            deduplication=dedup,
            decontamination=decontam,
            splits=splits,
            tokenizer=tok
        )
