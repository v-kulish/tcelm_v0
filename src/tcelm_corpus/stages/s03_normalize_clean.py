from typing import Dict, Any
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO
from ..normalize import TextNormalizer
from ..cleaners import (
    GenericCleaner, CCCCCleaner, WikimediaCleaner, StackExchangeCleaner,
    BooksCleaner, ScientificCleaner, EducationalCleaner, GovernmentLegalCleaner, TechnicalCleaner
)

class Stage03NormalizeClean(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("03_normalize_clean", output_dir, config)
        self.input_io = ParquetShardIO(f"{output_dir}/stages/02_select_pool")
        
        self.normalizer = TextNormalizer()
        self.generic_cleaner = GenericCleaner()
        self.cccc_cleaner = CCCCCleaner()
        self.wiki_cleaner = WikimediaCleaner()
        self.se_cleaner = StackExchangeCleaner()
        self.books_cleaner = BooksCleaner()
        self.sci_cleaner = ScientificCleaner()
        self.edu_cleaner = EducationalCleaner()
        self.gov_cleaner = GovernmentLegalCleaner()
        self.tech_cleaner = TechnicalCleaner()

    def run_stage(self) -> Dict[str, Any]:
        cleaned_records = []
        rejection_counts = {}
        record_counts = {}
        token_counts = {}

        source_cfg_map = {s.name: s for s in self.config.sources}
        all_input = list(self.input_io.read_shards())
        if not all_input:
            raise RuntimeError("Stage '03_normalize_clean' received 0 input records from Stage 02.")

        for rec in all_input:
            source_name = rec["source"]
            source_name_lower = source_name.lower()
            source_cfg = source_cfg_map.get(source_name)
            
            # 1. Normalization & PII Redaction
            norm_res = self.normalizer.normalize(rec["raw_text"])
            if norm_res.is_rejected:
                rejection_counts[norm_res.rejection_reason or "norm_rejected"] = rejection_counts.get(norm_res.rejection_reason or "norm_rejected", 0) + 1
                continue

            # 2. Generic Quality Cleaning
            min_doc_len = getattr(source_cfg, "min_doc_length", 128) if source_cfg else 128
            max_doc_len = getattr(source_cfg, "max_doc_length", 32768) if source_cfg else 32768
            min_eng = getattr(source_cfg, "min_english_prob", 0.50) if source_cfg else 0.50
            category = getattr(source_cfg, "category", "web") if source_cfg else "web"

            clean_res = self.generic_cleaner.clean(
                norm_res.normalized_text,
                source_category=category,
                min_doc_length=min_doc_len,
                max_doc_length=max_doc_len,
                min_english_prob=min_eng
            )
            if clean_res.is_rejected:
                rejection_counts[clean_res.rejection_reason or "generic_rejected"] = rejection_counts.get(clean_res.rejection_reason or "generic_rejected", 0) + 1
                continue

            cleaned_text = clean_res.cleaned_text

            # 3. Case-Insensitive Source Cleaner Dispatch
            if "cccc" in source_name_lower:
                sp_res = self.cccc_cleaner.clean(cleaned_text, rec.get("url", ""))
            elif "wikimedia" in source_name_lower:
                sp_res = self.wiki_cleaner.clean(cleaned_text, rec.get("title", ""))
            elif "stackexchange" in source_name_lower:
                sp_res = clean_res
            elif "gutenberg" in source_name_lower:
                sp_res = self.books_cleaner.clean_gutenberg(cleaned_text)
            elif "pre_1929" in source_name_lower:
                sp_res = self.books_cleaner.clean_pre1929(cleaned_text)
            elif "doab" in source_name_lower or "pressbooks" in source_name_lower:
                sp_res = self.books_cleaner.clean_doab_or_pressbooks(cleaned_text)
            elif "arxiv" in source_name_lower:
                sp_res = self.sci_cleaner.clean_arxiv(cleaned_text)
            elif "pes2o" in source_name_lower:
                sp_res = self.sci_cleaner.clean_pes2o(cleaned_text)
            elif "pubmed" in source_name_lower:
                sp_res = self.sci_cleaner.clean_pubmed(cleaned_text)
            elif "libretexts" in source_name_lower:
                sp_res = self.edu_cleaner.clean_libretexts(cleaned_text)
            elif "oercommons" in source_name_lower:
                sp_res = self.edu_cleaner.clean_oercommons(cleaned_text)
            elif "hansard" in source_name_lower:
                sp_res = clean_res
            elif "usgpo" in source_name_lower or "regulations" in source_name_lower or "caselaw" in source_name_lower:
                sp_res = self.gov_cleaner.clean_usgpo_regulations_caselaw(cleaned_text, source_type=source_name_lower)
            elif "pep" in source_name_lower or "python_enhancement" in source_name_lower:
                sp_res = self.tech_cleaner.clean_pep(cleaned_text)
            elif "github" in source_name_lower:
                sp_res = clean_res
            else:
                sp_res = clean_res

            if sp_res.is_rejected:
                rejection_counts[sp_res.rejection_reason or "source_rejected"] = rejection_counts.get(sp_res.rejection_reason or "source_rejected", 0) + 1
                continue

            rec_out = {
                "document_id": rec["document_id"],
                "parent_document_id": rec.get("parent_document_id", rec["document_id"]),
                "source": source_name,
                "priority": rec["priority"],
                "normalized_text": sp_res.cleaned_text,
                "license_status": rec.get("license_status", "missing"),
                "printable_ratio": clean_res.metrics.get("printable_ratio", 1.0),
                "alphabetic_ratio": clean_res.metrics.get("alphabetic_ratio", 1.0),
                "unique_line_ratio": clean_res.metrics.get("unique_line_ratio", 1.0),
                "pii_count": sum(norm_res.pii_counts.values()),
                "approx_tokens": len(sp_res.cleaned_text.split()),
                "title": rec.get("title", ""),
                "url": rec.get("url", ""),
                "domain": category
            }
            cleaned_records.append(rec_out)
            record_counts[source_name] = record_counts.get(source_name, 0) + 1
            token_counts[source_name] = token_counts.get(source_name, 0) + rec_out["approx_tokens"]

        written_shards = self.shard_io.write_records_to_shards(cleaned_records, shard_prefix="part")

        return {
            "record_counts": record_counts,
            "token_counts": token_counts,
            "rejection_counts": rejection_counts,
            "output_hashes": {"shard_count": len(written_shards)}
        }
