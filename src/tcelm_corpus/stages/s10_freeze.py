import os
import json
from typing import Dict, Any
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO
from ..storage.manifest import StageManifest
from ..tokenizer import BPECorpusTokenizer

class Stage10Freeze(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("10_freeze", output_dir, config)
        self.output_dir = output_dir
        self.tokenizer = BPECorpusTokenizer(
            vocab_size=self.config.tokenizer.vocab_size,
            special_tokens=self.config.tokenizer.special_tokens
        )

    def get_additional_cache_inputs(self) -> Dict[str, str]:
        tok_path = os.path.join(self.output_dir, "stages", "08_train_tokenizer", "tokenizer.json")
        if os.path.exists(tok_path):
            return {"tokenizer_sha256": StageManifest.compute_file_hash(tok_path)}
        return {}

    def run_stage(self) -> Dict[str, Any]:
        # 1. Require Stage 08 manifest and tokenizer SHA-256
        s08_man_path = os.path.join(self.output_dir, "stages", "08_train_tokenizer", "manifest.json")
        if not os.path.exists(s08_man_path):
            raise RuntimeError(f"Freeze Gate Failure: Stage 08 manifest missing at `{s08_man_path}`.")
        with open(s08_man_path, "r", encoding="utf-8") as f:
            s08_man = json.load(f)
            s08_tok_sha = s08_man.get("output_hashes", {}).get("tokenizer_sha256")
            if not s08_tok_sha:
                raise RuntimeError("Freeze Gate Failure: Stage 08 manifest missing `tokenizer_sha256`.")

        # 2. Require Stage 09 manifest and tokenizer SHA-256
        s09_man_path = os.path.join(self.output_dir, "stages", "09_tokenize_select", "manifest.json")
        if not os.path.exists(s09_man_path):
            raise RuntimeError(f"Freeze Gate Failure: Stage 09 manifest missing at `{s09_man_path}`.")
        with open(s09_man_path, "r", encoding="utf-8") as f:
            s09_man = json.load(f)
            s09_tok_sha = s09_man.get("output_hashes", {}).get("tokenizer_sha256")
            if not s09_tok_sha:
                raise RuntimeError("Freeze Gate Failure: Stage 09 manifest missing `tokenizer_sha256`.")

        # 3. Require current tokenizer file to exist and match Stage 08 & Stage 09 SHA-256
        tok_path = os.path.join(self.output_dir, "stages", "08_train_tokenizer", "tokenizer.json")
        if not os.path.exists(tok_path):
            raise RuntimeError(f"Freeze Gate Failure: Tokenizer file missing at `{tok_path}`.")

        current_tok_sha256 = StageManifest.compute_file_hash(tok_path)
        if not (s08_tok_sha == s09_tok_sha == current_tok_sha256):
            raise RuntimeError(
                f"Freeze Gate Failure: 3-way tokenizer SHA-256 mismatch: "
                f"Stage 08={s08_tok_sha[:10]}... vs Stage 09={s09_tok_sha[:10]}... vs Current={current_tok_sha256[:10]}..."
            )

        self.tokenizer.load_tokenizer(tok_path)
        if self.tokenizer.tokenizer is None:
            raise RuntimeError(f"Freeze Gate Failure: Failed loading tokenizer instance from `{tok_path}`.")

        stage_09_dir = os.path.join(self.output_dir, "stages", "09_tokenize_select")
        layer_a_dir = os.path.join(stage_09_dir, "layer_a_selected")
        layer_b_dir = os.path.join(stage_09_dir, "layer_b_selected")

        io_a = ParquetShardIO(layer_a_dir)
        io_b = ParquetShardIO(layer_b_dir)

        recs_a = list(io_a.read_shards())
        recs_b = list(io_b.read_shards())

        if not recs_a or not recs_b:
            raise RuntimeError("Stage '10_freeze' received 0 selected records from Stage 09.")

        docs_a = [r.get("document_id") or r.get("doc_id") for r in recs_a]
        docs_b = [r.get("document_id") or r.get("doc_id") for r in recs_b]

        if docs_a != docs_b:
            raise RuntimeError(f"Layer A and Layer B document ID mismatch in Stage 10 Freeze: Layer A has {len(docs_a)} docs, Layer B has {len(docs_b)} docs.")

        # Re-encoding verification gate: Layer A text must re-encode to exact Layer B token IDs with zero mismatch
        total_tokens = 0
        for ra, rb in zip(recs_a, recs_b):
            text_a = ra["normalized_text"]
            stored_ids_b = json.loads(rb["token_ids_json"])
            total_tokens += len(stored_ids_b)
            encoded_ids_a = self.tokenizer.tokenizer.encode(text_a).ids

            if encoded_ids_a != stored_ids_b:
                mismatch_idx = next(
                    (i for i, (a, b) in enumerate(zip(encoded_ids_a, stored_ids_b)) if a != b),
                    min(len(encoded_ids_a), len(stored_ids_b))
                )
                raise RuntimeError(
                    f"Freeze Gate Failure: Token ID sequence mismatch for doc `{ra['document_id']}` at index {mismatch_idx} "
                    f"(Encoded Layer A length={len(encoded_ids_a)} vs Stored Layer B length={len(stored_ids_b)})."
                )

        s08_man_sha256 = StageManifest.compute_file_hash(s08_man_path)
        s09_man_sha256 = StageManifest.compute_file_hash(s09_man_path)

        checksums = {
            "corpus_version": self.config.corpus_version,
            "stage_cache_key": self._compute_stage_cache_key(),
            "code_commit": self.code_identity,
            "tokenizer_sha256": current_tok_sha256,
            "stage_08_manifest_sha256": s08_man_sha256,
            "stage_09_manifest_sha256": s09_man_sha256,
            "tokenizer_vocab_size": self.config.tokenizer.vocab_size,
            "verified_documents": len(docs_a),
            "verified_tokens": total_tokens,
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

        print(f"Frozen corpus checksum manifest written to `{freeze_file}` ({len(docs_a):,} documents, {total_tokens:,} tokens verified 1-to-1).")
        return {
            "record_counts": {
                "verified_documents": len(docs_a),
                "verified_tokens": total_tokens,
                "layer_a_shards": len(checksums["layer_a_shards"]),
                "layer_b_shards": len(checksums["layer_b_shards"])
            },
            "output_hashes": {
                "freeze_manifest": freeze_file,
                "tokenizer_sha256": current_tok_sha256
            }
        }
