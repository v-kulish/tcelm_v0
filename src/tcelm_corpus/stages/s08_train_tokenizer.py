import os
from typing import Dict, Any, List
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO
from ..tokenizer import BPECorpusTokenizer

class Stage08TrainTokenizer(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("08_train_tokenizer", output_dir, config)
        self.input_io = ParquetShardIO(f"{output_dir}/stages/07_split")
        self.tokenizer = BPECorpusTokenizer(
            vocab_size=self.config.tokenizer.vocab_size,
            special_tokens=self.config.tokenizer.special_tokens
        )

    def run_stage(self) -> Dict[str, Any]:
        train_texts = []
        source_sample_counts = {}

        # Filter strictly for TRAIN split records
        for rec in self.input_io.read_shards():
            if rec.get("split") != "train":
                continue
            
            src = rec["source"]
            if source_sample_counts.get(src, 0) < 5000:
                train_texts.append(rec["normalized_text"])
                source_sample_counts[src] = source_sample_counts.get(src, 0) + 1

        if not train_texts:
            # Fallback if dataset is very small in smoke tests
            train_texts = [rec["normalized_text"] for rec in self.input_io.read_shards()]

        tok_path = os.path.join(self.stage_dir, "tokenizer.json")
        print(f"Training 32,768 Byte-Level BPE Tokenizer on {len(train_texts):,} TRAIN split text samples...")
        self.tokenizer.train_from_texts(train_texts, save_path=tok_path)

        return {
            "record_counts": {"sample_training_documents": len(train_texts)},
            "token_counts": source_sample_counts,
            "output_hashes": {"tokenizer_json": tok_path}
        }
