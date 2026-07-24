import re
import json
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
from tqdm import tqdm
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO

class Stage06Decontaminate(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("06_decontaminate", output_dir, config)
        self.input_io = ParquetShardIO(f"{output_dir}/stages/05_dedup")
        self.shingle_index = defaultdict(list)
        self.benchmark_items = []
        self._init_benchmark_registry()

    def _init_benchmark_registry(self):
        standard_benchmarks = [
            ("gsm8k", "gsm8k_0", "Janet has 16 apples. She gives 4 apples to her friend and keeps the rest."),
            ("humaneval", "he_0", "def has_close_elements(numbers: List[float], threshold: float) -> bool:"),
            ("mmlu", "mmlu_0", "Which of the following is a key property of prime numbers?")
        ]
        shingle_size = getattr(self.config.decontamination, "shingle_size", 13)

        for b_name, item_id, text in standard_benchmarks:
            words = re.findall(r'\w+', text.lower())
            if len(words) >= 5:
                shingles = {' '.join(words[i:i+shingle_size]) for i in range(max(1, len(words) - shingle_size + 1))}
                b_entry = {"b_name": b_name, "item_id": item_id, "text": text.lower(), "shingles": shingles}
                self.benchmark_items.append(b_entry)
                for sh in shingles:
                    self.shingle_index[sh].append(b_entry)

    def run_stage(self) -> Dict[str, Any]:
        all_input = list(self.input_io.read_shards())
        if not all_input:
            raise RuntimeError("Stage '06_decontaminate' received 0 input records from Stage 05.")

        retained_records = []
        contamination_logs = []
        contaminated_parents = set()

        shingle_size = getattr(self.config.decontamination, "shingle_size", 13)

        print(f"Stage 06 Decontamination: Scanning {len(all_input):,} documents against evaluation benchmark registry...")
        for rec in tqdm(all_input, desc="Decontaminating Documents", unit="doc"):
            doc_id = rec.get("document_id") or rec.get("doc_id")
            parent_id = rec.get("parent_document_id") or doc_id

            text_clean = rec["normalized_text"].lower()
            words = re.findall(r'\w+', text_clean)
            if not words:
                retained_records.append(rec)
                continue

            doc_shingles = {' '.join(words[i:i+shingle_size]) for i in range(max(1, len(words) - shingle_size + 1))}
            is_contaminated = False

            for b_item in self.benchmark_items:
                if b_item["text"] in text_clean:
                    is_contaminated = True
                    contaminated_parents.add(parent_id)
                    contamination_logs.append({
                        "benchmark_name": b_item["b_name"],
                        "item_id": b_item["item_id"],
                        "matched_document_id": doc_id,
                        "match_type": "exact_question_match",
                        "action": "removed_parent_family"
                    })
                    break

                shared = doc_shingles.intersection(b_item["shingles"])
                if shared:
                    overlap_ratio = len(shared) / max(len(b_item["shingles"]), 1)
                    if overlap_ratio >= getattr(self.config.decontamination, "min_token_overlap_ratio", 0.70):
                        is_contaminated = True
                        contaminated_parents.add(parent_id)
                        contamination_logs.append({
                            "benchmark_name": b_item["b_name"],
                            "item_id": b_item["item_id"],
                            "matched_document_id": doc_id,
                            "match_type": f"shingle_overlap_{overlap_ratio:.2f}",
                            "action": "removed_parent_family"
                        })
                        break

            if not is_contaminated:
                retained_records.append(rec)

        clean_records = [rec for rec in retained_records if (rec.get("parent_document_id") or rec.get("document_id")) not in contaminated_parents]
        written_shards = self.shard_io.write_records_to_shards(clean_records, shard_prefix="part")

        log_file = f"{self.stage_dir}/contamination_logs.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(contamination_logs, f, indent=2)

        record_counts = {}
        token_counts = {}
        for rec in clean_records:
            src = rec["source"]
            record_counts[src] = record_counts.get(src, 0) + 1
            token_counts[src] = token_counts.get(src, 0) + len(rec["normalized_text"].split())

        print(f"Stage 06 Decontamination complete: Retained {len(clean_records):,} / {len(all_input):,} documents ({len(contamination_logs):,} contaminated items removed).")

        return {
            "record_counts": record_counts,
            "token_counts": token_counts,
            "rejection_counts": {"contaminated_documents_removed": len(contamination_logs)},
            "output_hashes": {"shard_count": len(written_shards), "contamination_logs": log_file}
        }
