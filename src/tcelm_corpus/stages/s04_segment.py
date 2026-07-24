import json
from typing import Dict, Any
from tqdm import tqdm
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO
from ..segmentation import StructuralSegmenter
from ..schema import QualityScores

class Stage04Segment(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("04_segment", output_dir, config)
        self.input_io = ParquetShardIO(f"{output_dir}/stages/03_normalize_clean")
        self.segmenter = StructuralSegmenter()

    def run_stage(self) -> Dict[str, Any]:
        segmented_records = []
        record_counts = {}
        token_counts = {}

        all_input = list(self.input_io.read_shards())
        if not all_input:
            raise RuntimeError("Stage '04_segment' received 0 input records from Stage 03.")

        print(f"Stage 04 Structural Segmentation: Segmenting {len(all_input):,} cleaned documents...")
        for rec in tqdm(all_input, desc="Segmenting Documents", unit="doc"):
            quality = QualityScores(
                language_probability=1.0,
                printable_character_ratio=rec.get("printable_ratio", 1.0),
                alphabetic_character_ratio=rec.get("alphabetic_ratio", 1.0),
                repetition_ratio=1.0 - rec.get("unique_line_ratio", 1.0),
                pii_count=rec.get("pii_count", 0),
                ocr_quality=1.0,
                source_specific_quality=1.0,
                final_quality_score=1.0
            )

            metadata = {
                "title": rec.get("title", ""),
                "url": rec.get("url", ""),
                "license": rec.get("license_status", "missing"),
                "domain": rec.get("domain", "general")
            }

            doc_id = rec.get("document_id") or rec.get("doc_id", "doc")
            parent_id = rec.get("parent_document_id") or doc_id
            split_group_id = rec.get("split_group_id") or parent_id

            cdocs = self.segmenter.segment_document(
                doc_id=doc_id,
                parent_doc_id=parent_id,
                source=rec["source"],
                normalized_text=rec["normalized_text"],
                metadata=metadata,
                quality=quality
            )

            for cdoc in cdocs:
                cdoc.split_group_id = split_group_id
                rec_dict = cdoc.to_dict()
                
                # Convert dataclass fields to JSON strings for Parquet compatibility
                rec_dict["structure_json"] = json.dumps(rec_dict["structure"])
                rec_dict["quality_json"] = json.dumps(rec_dict["quality"])
                rec_dict["position_json"] = json.dumps(rec_dict["position"])
                rec_dict["priority"] = rec["priority"]
                rec_dict["split_group_id"] = split_group_id
                
                # Remove nested dict fields before Parquet serialization
                del rec_dict["structure"]
                del rec_dict["quality"]
                del rec_dict["position"]

                segmented_records.append(rec_dict)

                src = rec["source"]
                record_counts[src] = record_counts.get(src, 0) + 1
                token_counts[src] = token_counts.get(src, 0) + len(cdoc.normalized_text.split())

        written_shards = self.shard_io.write_records_to_shards(segmented_records, shard_prefix="part")

        print(f"Stage 04 Structural Segmentation complete: Produced {len(segmented_records):,} segments from {len(all_input):,} parent documents.")

        return {
            "record_counts": record_counts,
            "token_counts": token_counts,
            "output_hashes": {"shard_count": len(written_shards)}
        }
