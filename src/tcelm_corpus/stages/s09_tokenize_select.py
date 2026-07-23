import os
import json
from typing import Dict, Any, List
from collections import defaultdict
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO
from ..tokenizer import BPECorpusTokenizer
from ..schema import CanonicalDocument, StructureSpans

class Stage09TokenizeSelect(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("09_tokenize_select", output_dir, config)
        self.input_io = ParquetShardIO(f"{output_dir}/stages/07_split")
        self.tokenizer = BPECorpusTokenizer(
            vocab_size=self.config.tokenizer.vocab_size,
            special_tokens=self.config.tokenizer.special_tokens
        )
        tok_path = os.path.join(output_dir, "stages", "08_train_tokenizer", "tokenizer.json")
        if os.path.exists(tok_path):
            self.tokenizer.load_tokenizer(tok_path)

    def run_stage(self) -> Dict[str, Any]:
        source_records = defaultdict(list)
        
        # Read all split canonical documents
        for rec in self.input_io.read_shards():
            source_records[rec["source"]].append(rec)

        tokenized_selected_records = []
        record_counts = {}
        token_counts = {}

        for source_cfg in self.config.sources:
            source_name = source_cfg.name
            target_quota = self.config.get_source_quota(source_name)

            recs = source_records.get(source_name, [])
            # Sort by deterministic priority q(d)
            recs.sort(key=lambda x: x["priority"])

            accumulated_tokens = 0
            accumulated_docs = 0

            for rec in recs:
                # Reconstruct CanonicalDocument to encode with offsets
                struct_dict = json.loads(rec["structure_json"]) if isinstance(rec.get("structure_json"), str) else rec.get("structure_json", {})
                spans = StructureSpans(**struct_dict) if isinstance(struct_dict, dict) else StructureSpans()
                
                cdoc = CanonicalDocument(
                    document_id=rec["doc_id"],
                    parent_document_id=rec["parent_document_id"],
                    source=rec["source"],
                    source_revision=rec.get("source_revision", "v0.1"),
                    source_record_id=rec.get("source_record_id", rec["doc_id"]),
                    source_url_or_provenance=rec.get("url", ""),
                    license=rec.get("license", "open"),
                    authors=rec.get("authors", ""),
                    title=rec.get("title", ""),
                    publication_date=rec.get("publication_date", ""),
                    language="en",
                    raw_text_hash=rec.get("raw_text_hash", ""),
                    normalized_text_hash=rec.get("normalized_text_hash", ""),
                    dedup_cluster_id=rec.get("dedup_cluster_id", rec["doc_id"]),
                    normalized_text=rec["normalized_text"],
                    document_type=rec.get("document_type", "article"),
                    domain=rec.get("domain", "general"),
                    genre=rec.get("genre", "prose"),
                    structure=spans,
                    split=rec.get("split", "train")
                )

                tdoc = self.tokenizer.encode_document(cdoc)
                tok_len = len(tdoc.token_ids)

                # Post-tokenization quota enforcement check
                if accumulated_tokens + tok_len > target_quota and accumulated_tokens > 0:
                    # Final segment sentence/paragraph-aligned truncation
                    needed = target_quota - accumulated_tokens
                    if needed > 64 and tdoc.paragraph_token_spans:
                        # Truncate at nearest paragraph boundary
                        cut_idx = tok_len
                        for p_start, p_end in tdoc.paragraph_token_spans:
                            if p_end <= needed:
                                cut_idx = p_end
                        tdoc.token_ids = tdoc.token_ids[:cut_idx]
                        tok_len = cut_idx

                rec_out = {
                    "document_id": tdoc.document_id,
                    "parent_document_id": tdoc.parent_document_id,
                    "source": tdoc.source,
                    "split": tdoc.split,
                    "token_ids_json": json.dumps(tdoc.token_ids),
                    "sentence_spans_json": json.dumps(tdoc.sentence_token_spans),
                    "paragraph_spans_json": json.dumps(tdoc.paragraph_token_spans),
                    "turn_spans_json": json.dumps(tdoc.turn_token_spans),
                    "equation_spans_json": json.dumps(tdoc.equation_token_spans),
                    "token_count": tok_len,
                    "priority": rec["priority"]
                }
                tokenized_selected_records.append(rec_out)
                accumulated_tokens += tok_len
                accumulated_docs += 1

                if accumulated_tokens >= target_quota:
                    break

            record_counts[source_name] = accumulated_docs
            token_counts[source_name] = accumulated_tokens
            print(f"Source `{source_name}` final post-tokenization quota: {accumulated_docs:,} docs, {accumulated_tokens:,} tokens (Target: {target_quota:,}).")

        written_shards = self.shard_io.write_records_to_shards(tokenized_selected_records, shard_prefix="layer_b")

        return {
            "record_counts": record_counts,
            "token_counts": token_counts,
            "output_hashes": {"shard_count": len(written_shards)}
        }
