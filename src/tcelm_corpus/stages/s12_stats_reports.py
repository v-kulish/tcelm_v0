import os
import json
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO
from ..storage.manifest import StageManifest
from ..stats import TokenFrequencyStats
from ..reports import MandatoryReportGenerator
from ..schema import CanonicalDocument, TokenizedDocument, StructureSpans, QualityScores

def sanitize_npz_key(source_name: str) -> str:
    return source_name.replace("/", "_").replace("-", "_").replace(".", "_")

class Stage12StatsReports(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("12_stats_reports", output_dir, config)
        self.output_dir = output_dir
        stage_09_dir = os.path.join(output_dir, "stages", "09_tokenize_select")
        self.layer_a_io = ParquetShardIO(os.path.join(stage_09_dir, "layer_a_selected"))
        self.layer_b_io = ParquetShardIO(os.path.join(stage_09_dir, "layer_b_selected"))
        self.decontam_io = ParquetShardIO(f"{output_dir}/stages/06_decontaminate")
        self.stats_calc = TokenFrequencyStats(vocab_size=self.config.tokenizer.vocab_size)
        self.report_gen = MandatoryReportGenerator(os.path.join(output_dir, "reports"))

    def get_additional_cache_inputs(self) -> Dict[str, str]:
        tok_path = os.path.join(self.output_dir, "stages", "08_train_tokenizer", "tokenizer.json")
        freeze_path = os.path.join(self.output_dir, "stages", "10_freeze", "freeze_manifest.json")
        inputs = {}
        if os.path.exists(tok_path):
            inputs["tokenizer_sha256"] = StageManifest.compute_file_hash(tok_path)
        if os.path.exists(freeze_path):
            inputs["freeze_manifest_sha256"] = StageManifest.compute_file_hash(freeze_path)
        return inputs

    def run_stage(self) -> Dict[str, Any]:
        s08_man_path = os.path.join(self.output_dir, "stages", "08_train_tokenizer", "manifest.json")
        if not os.path.exists(s08_man_path):
            raise RuntimeError(f"Frequency Metadata Failure: Stage 08 manifest missing at `{s08_man_path}`.")
        with open(s08_man_path, "r", encoding="utf-8") as f:
            s08_man = json.load(f)
            s08_tok_sha = s08_man.get("output_hashes", {}).get("tokenizer_sha256")
            if not s08_tok_sha:
                raise RuntimeError("Frequency Metadata Failure: Stage 08 manifest missing `tokenizer_sha256`.")

        freeze_path = os.path.join(self.output_dir, "stages", "10_freeze", "freeze_manifest.json")
        if not os.path.exists(freeze_path):
            raise RuntimeError(f"Frequency Metadata Failure: Stage 10 freeze manifest missing at `{freeze_path}`.")
        with open(freeze_path, "r", encoding="utf-8") as f:
            freeze_man = json.load(f)
            freeze_tok_sha = freeze_man.get("tokenizer_sha256")
            if not freeze_tok_sha:
                raise RuntimeError("Frequency Metadata Failure: Freeze manifest missing `tokenizer_sha256`.")

        tok_path = os.path.join(self.output_dir, "stages", "08_train_tokenizer", "tokenizer.json")
        if not os.path.exists(tok_path):
            raise RuntimeError(f"Frequency Metadata Failure: Tokenizer file missing at `{tok_path}`.")

        current_tok_sha = StageManifest.compute_file_hash(tok_path)
        if not (s08_tok_sha == freeze_tok_sha == current_tok_sha):
            raise RuntimeError(
                f"Frequency Metadata Failure: Tokenizer SHA-256 mismatch: "
                f"Stage 08={s08_tok_sha[:10]}... vs Freeze={freeze_tok_sha[:10]}... vs Current={current_tok_sha[:10]}..."
            )

        all_a = list(self.layer_a_io.read_shards())
        all_b = list(self.layer_b_io.read_shards())

        if not all_a or not all_b:
            raise RuntimeError("Stage '12_stats_reports' received 0 selected Layer A or Layer B records from Stage 09.")

        canonical_docs: List[CanonicalDocument] = []
        for rec in all_a:
            doc_id = rec.get("document_id") or rec.get("doc_id")
            parent_id = rec.get("parent_document_id") or doc_id

            struct_dict = json.loads(rec["structure_json"]) if isinstance(rec.get("structure_json"), str) else rec.get("structure_json", {})
            spans = StructureSpans(**struct_dict) if isinstance(struct_dict, dict) else StructureSpans()

            cdoc = CanonicalDocument(
                document_id=doc_id,
                parent_document_id=parent_id,
                source=rec["source"],
                source_revision=rec.get("source_revision", "v0.1"),
                source_record_id=rec.get("source_record_id", doc_id),
                source_url_or_provenance=rec.get("url", ""),
                license=rec.get("license_status", "missing"),
                authors=rec.get("authors", ""),
                title=rec.get("title", ""),
                publication_date="",
                language="en",
                raw_text_hash=rec.get("raw_text_hash", ""),
                normalized_text_hash=rec.get("normalized_text_hash", ""),
                dedup_cluster_id=rec.get("dedup_cluster_id", doc_id),
                normalized_text=rec["normalized_text"],
                document_type="article",
                domain=rec.get("domain", "web"),
                genre="prose",
                structure=spans,
                quality=QualityScores(
                    printable_character_ratio=rec.get("printable_ratio", 1.0),
                    alphabetic_character_ratio=rec.get("alphabetic_ratio", 1.0),
                    pii_count=rec.get("pii_count", 0)
                ),
                split=rec.get("split", "train")
            )
            canonical_docs.append(cdoc)

        tokenized_docs: List[TokenizedDocument] = []
        source_train_tokens = defaultdict(int)

        for rec in all_b:
            doc_id = rec.get("document_id") or rec.get("doc_id")
            parent_id = rec.get("parent_document_id") or doc_id

            tok_ids = json.loads(rec["token_ids_json"])
            tdoc = TokenizedDocument(
                document_id=doc_id,
                parent_document_id=parent_id,
                source=rec["source"],
                split=rec["split"],
                token_ids=tok_ids
            )
            tokenized_docs.append(tdoc)

            if rec["split"] == "train":
                source_train_tokens[rec["source"]] += len(tok_ids)

        # 1. Compute smoothed unigram log probabilities on TRAIN split only
        print("Computing unigram frequency log probabilities on TRAIN split...")
        freq_stats = self.stats_calc.compute_frequencies(tokenized_docs)

        p_global_arr = freq_stats["p_global_log"]
        p_source_dict = freq_stats["p_source_log"]
        total_train_tokens = freq_stats["total_train_tokens"]

        # Validate token count sum & source set consistency
        sum_source_tokens = sum(source_train_tokens.values())
        if sum_source_tokens != total_train_tokens:
            raise RuntimeError(f"Frequency Consistency Failure: Sum of per-source train tokens ({sum_source_tokens:,}) does not equal global train tokens ({total_train_tokens:,}).")

        if set(source_train_tokens.keys()) != set(p_source_dict.keys()):
            raise RuntimeError(f"Frequency Consistency Failure: Per-source token sources ({set(source_train_tokens.keys())}) do not match frequency stats sources ({set(p_source_dict.keys())}).")

        # Validate array shapes, float32 dtype, and finite values
        expected_shape = (self.config.tokenizer.vocab_size,)
        if p_global_arr.shape != expected_shape or p_global_arr.dtype != np.float32 or not np.all(np.isfinite(p_global_arr)):
            raise RuntimeError(f"Frequency Validation Failure: Global array invalid (shape={p_global_arr.shape}, dtype={p_global_arr.dtype}).")

        for src, s_arr in p_source_dict.items():
            if s_arr.shape != expected_shape or s_arr.dtype != np.float32 or not np.all(np.isfinite(s_arr)):
                raise RuntimeError(f"Frequency Validation Failure: Source array `{src}` invalid (shape={s_arr.shape}, dtype={s_arr.dtype}).")

        # PERSIST binary float32 numpy arrays (.npy & .npz) for fast training loaders
        global_unigram_npy = os.path.join(self.stage_dir, "unigram_log_probs.npy")
        np.save(global_unigram_npy, p_global_arr)

        source_unigram_npz = os.path.join(self.stage_dir, "source_unigram_log_probs.npz")
        np.savez(source_unigram_npz, **{sanitize_npz_key(k): v for k, v in p_source_dict.items()})

        # Also persist JSON files for inspection
        global_unigram_json = os.path.join(self.stage_dir, "unigram_log_probs.json")
        with open(global_unigram_json, "w", encoding="utf-8") as f:
            json.dump(p_global_arr.tolist(), f)

        source_unigram_json = os.path.join(self.stage_dir, "source_unigram_log_probs.json")
        source_log_probs_serialized = {src: arr.tolist() for src, arr in p_source_dict.items()}
        with open(source_unigram_json, "w", encoding="utf-8") as f:
            json.dump(source_log_probs_serialized, f)

        freeze_sha256 = StageManifest.compute_file_hash(freeze_path)
        source_keys_map = {sanitize_npz_key(src): src for src in p_source_dict.keys()}

        # Generate Frequency Companion Metadata Manifest
        freq_meta = {
            "schema_version": 1,
            "corpus_version": self.config.corpus_version,
            "dtype": "float32",
            "vocab_size": self.config.tokenizer.vocab_size,
            "smoothing_alpha": self.stats_calc.smoothing_alpha,
            "total_train_tokens": total_train_tokens,
            "total_train_documents": len([d for d in tokenized_docs if d.split == "train"]),
            "tokenizer_sha256": current_tok_sha,
            "freeze_manifest_sha256": freeze_sha256,
            "global_unigram_file": "unigram_log_probs.npy",
            "global_unigram_shape": list(p_global_arr.shape),
            "global_unigram_npy_sha256": StageManifest.compute_file_hash(global_unigram_npy),
            "source_unigram_file": "source_unigram_log_probs.npz",
            "source_unigram_npz_sha256": StageManifest.compute_file_hash(source_unigram_npz),
            "source_keys": source_keys_map,
            "source_train_tokens": dict(source_train_tokens)
        }

        freq_meta_path = os.path.join(self.stage_dir, "frequency_metadata.json")
        with open(freq_meta_path, "w", encoding="utf-8") as f:
            json.dump(freq_meta, f, indent=2)

        # 2. Read contamination logs if present
        decontam_manifest = os.path.join(self.output_dir, "stages", "06_decontaminate", "contamination_logs.json")
        contamination_logs = []
        if os.path.exists(decontam_manifest):
            with open(decontam_manifest, "r", encoding="utf-8") as f:
                contamination_logs = json.load(f)

        # 3. Read stage manifests for rejection counts
        s03_manifest = os.path.join(self.output_dir, "stages", "03_normalize_clean", "manifest.json")
        s03_rejections = {}
        if os.path.exists(s03_manifest):
            with open(s03_manifest, "r", encoding="utf-8") as f:
                s03_rejections = json.load(f).get("rejection_counts", {})

        s05_manifest = os.path.join(self.output_dir, "stages", "05_dedup", "manifest.json")
        s05_stats = {}
        if os.path.exists(s05_manifest):
            with open(s05_manifest, "r", encoding="utf-8") as f:
                s05_stats = json.load(f).get("rejection_counts", {})

        # 4. Generate 7 Mandatory Pre-training Audit Reports
        report_files = self.report_gen.generate_all_reports(
            canonical_docs,
            tokenized_docs,
            rejection_counts=s03_rejections,
            contamination_logs=contamination_logs,
            dedup_stats=s05_stats
        )

        output_artifacts = dict(report_files)
        output_artifacts["frequency_metadata_json"] = freq_meta_path
        output_artifacts["global_unigram_log_probs_npy"] = global_unigram_npy
        output_artifacts["source_unigram_log_probs_npz"] = source_unigram_npz
        output_artifacts["global_unigram_log_probs_json"] = global_unigram_json
        output_artifacts["source_unigram_log_probs_json"] = source_unigram_json

        return {
            "record_counts": {
                "canonical_documents": len(canonical_docs),
                "tokenized_documents": len(tokenized_docs)
            },
            "output_hashes": output_artifacts
        }
