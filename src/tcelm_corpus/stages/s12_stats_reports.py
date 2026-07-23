import os
import json
from typing import Dict, Any, List
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO
from ..stats import TokenFrequencyStats
from ..reports import MandatoryReportGenerator
from ..schema import CanonicalDocument, TokenizedDocument, StructureSpans, QualityScores

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

    def run_stage(self) -> Dict[str, Any]:
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

        # 1. Compute smoothed unigram log probabilities on TRAIN split only
        print("Computing unigram frequency log probabilities on TRAIN split...")
        freq_stats = self.stats_calc.compute_frequencies(tokenized_docs)

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

        return {
            "record_counts": {
                "canonical_documents": len(canonical_docs),
                "tokenized_documents": len(tokenized_docs)
            },
            "output_hashes": report_files
        }
