import os
import re
import json
import hashlib
import struct
from typing import Dict, Any, List
from datasets import load_dataset
from huggingface_hub import HfApi
from .base_stage import BaseStage
from ..adapters import get_source_adapter

SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

def compute_priority(corpus_version: str, source: str, doc_id: str, seed: int = 42) -> int:
    key = f"{corpus_version}:{source}:{doc_id}:{seed}".encode('utf-8')
    digest = hashlib.sha256(key).digest()[:8]
    return struct.unpack(">Q", digest)[0] & 0x7FFFFFFFFFFFFFFF

class Stage01Ingest(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("01_ingest", output_dir, config)
        self._resolved_shas_cache = None

    def _resolve_source_shas(self) -> Dict[str, Any]:
        if self._resolved_shas_cache is not None:
            return self._resolved_shas_cache

        resolved_shas = {}
        api = HfApi()
        for source_cfg in self.config.sources:
            source_name = source_cfg.name
            requested_rev = getattr(source_cfg, "revision", "main") or "main"
            
            # 1. If requested revision is already an immutable 40-character hex SHA
            if SHA_RE.fullmatch(requested_rev):
                resolved_shas[source_name] = {
                    "requested_revision": requested_rev,
                    "resolved_sha": requested_rev,
                    "status": "pinned_sha"
                }
                continue

            # 2. Query Hugging Face Hub API for dataset info commit SHA
            try:
                info = api.dataset_info(repo_id=source_name, revision=requested_rev)
                sha = getattr(info, "sha", None)
                if sha and SHA_RE.fullmatch(sha):
                    resolved_shas[source_name] = {
                        "requested_revision": requested_rev,
                        "resolved_sha": sha,
                        "status": "api_resolved"
                    }
                else:
                    if getattr(self.config, "production_mode", False):
                        raise RuntimeError(f"Production Build Failure: Invalid or absent commit SHA ({sha}) returned for source `{source_name}` (Revision `{requested_rev}`).")
                    resolved_shas[source_name] = {
                        "requested_revision": requested_rev,
                        "resolved_sha": None,
                        "status": "unresolved_fallback"
                    }
            except Exception as e:
                if getattr(self.config, "production_mode", False):
                    raise RuntimeError(f"Production Build Failure: Failed resolving commit SHA for source `{source_name}` (Revision `{requested_rev}`): {e}")
                resolved_shas[source_name] = {
                    "requested_revision": requested_rev,
                    "resolved_sha": None,
                    "status": "unresolved_fallback"
                }

        self._resolved_shas_cache = resolved_shas
        return resolved_shas

    def get_additional_cache_inputs(self) -> Dict[str, str]:
        shas = self._resolve_source_shas()
        digest = hashlib.sha256(json.dumps(shas, sort_keys=True).encode("utf-8")).hexdigest()
        return {"resolved_sources_sha256": digest}

    def run_stage(self) -> Dict[str, Any]:
        record_counts = {}
        token_counts = {}
        oversized_skipped_counts = {}
        all_shards = []

        smoke_total = getattr(self.config, "smoke_total_tokens", None)
        resolved_info = self._resolve_source_shas()

        for source_cfg in self.config.sources:
            source_name = source_cfg.name
            rev_data = resolved_info.get(source_name, {})
            requested_revision = rev_data.get("requested_revision", "main")
            resolved_revision_sha = rev_data.get("resolved_sha") or requested_revision
            resolution_status = rev_data.get("status", "unresolved_fallback")

            print(f"Ingesting `{source_name}` (Requested: `{requested_revision}`, Resolved SHA: `{resolved_revision_sha[:10] if len(resolved_revision_sha)>=10 else resolved_revision_sha}`, Status: `{resolution_status}`)...")

            adapter = get_source_adapter(source_name, source_cfg.__dict__)
            
            try:
                ds = load_dataset(source_name, split="train", streaming=True, revision=resolved_revision_sha)
            except Exception as e:
                if getattr(self.config, "production_mode", False):
                    raise RuntimeError(f"Production Build Failure: Failed streaming source `{source_name}` (Revision: `{resolved_revision_sha}`): {e}")
                print(f"Warning: Failed streaming `{source_name}` with revision `{resolved_revision_sha}`: {e}")
                continue

            records_batch = []
            source_doc_count = 0
            source_token_count = 0
            oversized_skipped_doc_count = 0

            # Proportional smoke token budget & strict admission limit calculation
            source_smoke_budget = None
            max_single_doc_prov_tokens = None
            if smoke_total is not None:
                source_ratio = getattr(source_cfg, "share", getattr(source_cfg, "target_ratio", 0.0))
                source_smoke_budget = int(smoke_total * source_ratio * 1.35)
                max_single_doc_prov_tokens = min(source_smoke_budget, max(2048, int(source_smoke_budget * 0.5)))

            # Stream candidates
            for i, raw_item in enumerate(ds):
                raw_id = str(raw_item.get("id", raw_item.get("doc_id", f"{source_name}_{i}")))
                raw_text, license_status, metadata = adapter.extract_record(raw_item)
                
                if not raw_text or not raw_text.strip():
                    continue

                prov_tokens = len(raw_text.split())

                is_structural_probe = False
                is_eligible_balanced = True

                # Strict non-destructive smoke admission check
                if max_single_doc_prov_tokens is not None and prov_tokens > max_single_doc_prov_tokens:
                    if source_doc_count > 0:
                        oversized_skipped_doc_count += 1
                        continue
                    else:
                        # First document of long-form source -> Structural probe only (excluded from balanced pool)
                        is_structural_probe = True
                        is_eligible_balanced = False

                raw_text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
                document_id = hashlib.sha256(f"{source_name}:{resolved_revision_sha}:{raw_id}:{raw_text_hash}".encode("utf-8")).hexdigest()[:32]
                parent_document_id = document_id

                priority = compute_priority(self.config.corpus_version, source_name, raw_id, self.config.seed)

                rec = {
                    "document_id": document_id,
                    "parent_document_id": parent_document_id,
                    "split_group_id": parent_document_id,
                    "source_repository": source_name,
                    "requested_source_revision": requested_revision,
                    "resolved_source_revision_sha": resolved_revision_sha,
                    "revision_resolution_status": resolution_status,
                    "source_record_id": raw_id,
                    "source_url_or_provenance": metadata.get("url", ""),
                    "source": source_name,
                    "license": license_status,
                    "raw_text_hash": raw_text_hash,
                    "priority": priority,
                    "provisional_tokens": prov_tokens,
                    "raw_text": raw_text,
                    "license_status": license_status,
                    "eligible_for_balanced_pool": is_eligible_balanced,
                    "smoke_structural_probe": is_structural_probe,
                    "title": metadata.get("title", ""),
                    "url": metadata.get("url", ""),
                    "metadata_json": str(metadata)
                }
                records_batch.append(rec)
                
                if is_eligible_balanced:
                    source_doc_count += 1
                    source_token_count += prov_tokens

                if len(records_batch) >= 5000:
                    shards = self.shard_io.write_records_to_shards(records_batch, shard_prefix="part")
                    all_shards.extend(shards)
                    records_batch = []

                if getattr(self.config, "max_records_per_source", None) and source_doc_count >= self.config.max_records_per_source:
                    break

                if source_smoke_budget is not None and source_token_count >= source_smoke_budget:
                    print(f"Source `{source_name}` reached proportional smoke token budget ({source_token_count:,} >= {source_smoke_budget:,}).")
                    break

            if records_batch:
                shards = self.shard_io.write_records_to_shards(records_batch, shard_prefix="part")
                all_shards.extend(shards)

            if source_doc_count == 0 and not oversized_skipped_doc_count and getattr(self.config, "production_mode", False):
                raise RuntimeError(f"Production Build Failure: Source `{source_name}` yielded 0 records.")

            record_counts[source_name] = source_doc_count
            token_counts[source_name] = source_token_count
            oversized_skipped_counts[source_name] = oversized_skipped_doc_count

        return {
            "record_counts": record_counts,
            "token_counts": token_counts,
            "rejection_counts": {"oversized_documents_skipped": oversized_skipped_counts},
            "output_hashes": {
                "shard_count": len(all_shards),
                "resolved_source_revisions": resolved_info
            }
        }
