import os
import hashlib
import struct
from typing import Dict, Any, List
from datasets import load_dataset
from .base_stage import BaseStage
from ..adapters import get_source_adapter

def compute_priority(corpus_version: str, source: str, doc_id: str, seed: int = 42) -> int:
    key = f"{corpus_version}:{source}:{doc_id}:{seed}".encode('utf-8')
    digest = hashlib.sha256(key).digest()[:8]
    return struct.unpack(">Q", digest)[0] & 0x7FFFFFFFFFFFFFFF

class Stage01Ingest(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("01_ingest", output_dir, config)

    def run_stage(self) -> Dict[str, Any]:
        record_counts = {}
        token_counts = {}
        all_shards = []

        for source_cfg in self.config.sources:
            source_name = source_cfg.name
            revision = getattr(source_cfg, "revision", "main") or "main"
            print(f"Ingesting `{source_name}` (Revision: `{revision}`)...")

            adapter = get_source_adapter(source_name, source_cfg.__dict__)
            
            try:
                ds = load_dataset(source_name, split="train", streaming=True, revision=revision)
            except Exception as e:
                if getattr(self.config, "production_mode", False):
                    raise RuntimeError(f"Production Build Failure: Failed streaming source `{source_name}` (Revision: `{revision}`): {e}")
                print(f"Warning: Failed streaming `{source_name}` with revision `{revision}`: {e}")
                continue

            records_batch = []
            source_doc_count = 0
            source_token_count = 0

            # Stream candidates
            for i, raw_item in enumerate(ds):
                raw_id = str(raw_item.get("id", raw_item.get("doc_id", f"{source_name}_{i}")))
                document_id = hashlib.sha256(f"{source_name}:{raw_id}".encode("utf-8")).hexdigest()[:32]
                parent_document_id = document_id

                raw_text, license_status, metadata = adapter.extract_record(raw_item)
                
                if not raw_text or not raw_text.strip():
                    continue

                priority = compute_priority(self.config.corpus_version, source_name, raw_id, self.config.seed)
                prov_tokens = len(raw_text.split())

                rec = {
                    "document_id": document_id,
                    "parent_document_id": parent_document_id,
                    "split_group_id": parent_document_id,
                    "source_record_id": raw_id,
                    "source": source_name,
                    "priority": priority,
                    "provisional_tokens": prov_tokens,
                    "raw_text": raw_text,
                    "license_status": license_status,
                    "title": metadata.get("title", ""),
                    "url": metadata.get("url", ""),
                    "metadata_json": str(metadata)
                }
                records_batch.append(rec)
                source_doc_count += 1
                source_token_count += prov_tokens

                if len(records_batch) >= 5000:
                    shards = self.shard_io.write_records_to_shards(records_batch, shard_prefix="part")
                    all_shards.extend(shards)
                    records_batch = []

                if getattr(self.config, "max_records_per_source", None) and source_doc_count >= self.config.max_records_per_source:
                    break

            if records_batch:
                shards = self.shard_io.write_records_to_shards(records_batch, shard_prefix="part")
                all_shards.extend(shards)

            if source_doc_count == 0 and getattr(self.config, "production_mode", False):
                raise RuntimeError(f"Production Build Failure: Source `{source_name}` yielded 0 records.")

            record_counts[source_name] = source_doc_count
            token_counts[source_name] = source_token_count

        return {
            "record_counts": record_counts,
            "token_counts": token_counts,
            "output_hashes": {"shard_count": len(all_shards)}
        }
