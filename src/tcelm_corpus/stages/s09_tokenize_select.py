import os
import json
import hashlib
from typing import Dict, Any, List
from collections import defaultdict
from tqdm import tqdm
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO
from ..storage.manifest import StageManifest
from ..tokenizer import BPECorpusTokenizer
from ..schema import CanonicalDocument, StructureSpans
from ..segmentation import StructuralSegmenter

def trim_spans(spans: List[List[int]], max_len: int) -> List[List[int]]:
    result = []
    for s_start, s_end in spans:
        if s_start < max_len:
            result.append([s_start, min(s_end, max_len)])
    return result

class Stage09TokenizeSelect(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("09_tokenize_select", output_dir, config)
        self.output_dir = output_dir
        self.input_io = ParquetShardIO(f"{output_dir}/stages/07_split")
        self.layer_a_out_io = ParquetShardIO(f"{self.stage_dir}/layer_a_selected")
        self.layer_b_out_io = ParquetShardIO(f"{self.stage_dir}/layer_b_selected")
        self.tokenizer = BPECorpusTokenizer(
            vocab_size=self.config.tokenizer.vocab_size,
            special_tokens=self.config.tokenizer.special_tokens
        )
        self.segmenter = StructuralSegmenter()

    def get_additional_cache_inputs(self) -> Dict[str, str]:
        tok_path = os.path.join(self.output_dir, "stages", "08_train_tokenizer", "tokenizer.json")
        if os.path.exists(tok_path):
            return {"tokenizer_sha256": StageManifest.compute_file_hash(tok_path)}
        return {}

    def run_stage(self) -> Dict[str, Any]:
        s08_man_path = os.path.join(self.output_dir, "stages", "08_train_tokenizer", "manifest.json")
        if not os.path.exists(s08_man_path):
            raise RuntimeError(f"Tokenizer Provenance Failure: Stage 08 manifest missing at `{s08_man_path}`.")

        with open(s08_man_path, "r", encoding="utf-8") as f:
            s08_man = json.load(f)
            s08_tok_sha = s08_man.get("output_hashes", {}).get("tokenizer_sha256")
            if not s08_tok_sha:
                raise RuntimeError("Tokenizer Provenance Failure: Stage 08 manifest missing `tokenizer_sha256`.")

        tok_path = os.path.join(self.output_dir, "stages", "08_train_tokenizer", "tokenizer.json")
        if not os.path.exists(tok_path):
            raise RuntimeError(f"Tokenizer file not found at `{tok_path}` in Stage 09 TokenizeSelect.")

        tok_sha256 = StageManifest.compute_file_hash(tok_path)
        if tok_sha256 != s08_tok_sha:
            raise RuntimeError(f"Tokenizer Provenance Failure: Current tokenizer hash ({tok_sha256[:10]}...) does not match Stage 08 manifest ({s08_tok_sha[:10]}...).")

        self.tokenizer.load_tokenizer(tok_path)

        all_input = list(self.input_io.read_shards())
        if not all_input:
            raise RuntimeError("Stage '09_tokenize_select' received 0 input records from Stage 07.")

        source_records = defaultdict(list)
        for rec in all_input:
            doc_id = rec.get("document_id") or rec.get("doc_id")
            rec["document_id"] = doc_id
            source_records[rec["source"]].append(rec)

        tokenized_selected_records = []
        canonical_selected_records = []
        record_counts = {}
        token_counts = {}

        print(f"Stage 09 Tokenization & Quota Selection: Tokenizing records across {len(self.config.sources)} sources...")
        for source_cfg in tqdm(self.config.sources, desc="Tokenizing Sources", unit="source"):
            source_name = source_cfg.name
            target_quota = self.config.get_source_quota(source_name)

            recs = source_records.get(source_name, [])
            # Sort by deterministic priority q(d)
            recs.sort(key=lambda x: x["priority"])

            accumulated_tokens = 0
            accumulated_docs = 0

            for rec in recs:
                doc_id = rec["document_id"]
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
                    license=rec.get("license", "open"),
                    authors=rec.get("authors", ""),
                    title=rec.get("title", ""),
                    publication_date=rec.get("publication_date", ""),
                    language="en",
                    raw_text_hash=rec.get("raw_text_hash", ""),
                    normalized_text_hash=rec.get("normalized_text_hash", ""),
                    dedup_cluster_id=rec.get("dedup_cluster_id", doc_id),
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
                if accumulated_tokens + tok_len > target_quota:
                    needed = target_quota - accumulated_tokens
                    if needed <= 0:
                        break

                    cut_idx = tok_len
                    if tdoc.paragraph_token_spans:
                        for p_start, p_end in tdoc.paragraph_token_spans:
                            if p_end <= needed:
                                cut_idx = p_end
                    if cut_idx == tok_len and tdoc.sentence_token_spans:
                        for s_start, s_end in tdoc.sentence_token_spans:
                            if s_end <= needed:
                                cut_idx = s_end
                    if cut_idx == tok_len:
                        cut_idx = needed

                    if cut_idx < 64 and accumulated_tokens > 0:
                        # Omit segment if less than minimum length
                        break

                    # Decode cut tokens back to raw text, then re-encode text to ensure exact 1-to-1 token ID equivalence
                    truncated_text = self.tokenizer.tokenizer.decode(tdoc.token_ids[:cut_idx])
                    cdoc.normalized_text = truncated_text
                    cdoc.normalized_text_hash = hashlib.sha256(truncated_text.encode("utf-8")).hexdigest()
                    cdoc.structure = self.segmenter.extract_structure_spans(truncated_text)
                    rec["normalized_text"] = truncated_text
                    rec["normalized_text_hash"] = cdoc.normalized_text_hash
                    rec["structure_json"] = json.dumps(cdoc.structure.__dict__)

                    # Re-encode canonical document so Layer B token IDs match Layer A text 1-to-1 under tokenizer.encode()
                    tdoc = self.tokenizer.encode_document(cdoc)
                    tok_len = len(tdoc.token_ids)

                rec_b = {
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
                tokenized_selected_records.append(rec_b)
                canonical_selected_records.append(rec)

                accumulated_tokens += tok_len
                accumulated_docs += 1

                if accumulated_tokens >= target_quota:
                    break

            if getattr(self.config, "production_mode", False):
                diff_ratio = abs(accumulated_tokens - target_quota) / max(target_quota, 1)
                if diff_ratio > 0.0005:
                    raise RuntimeError(f"Production Build Failure: Source `{source_name}` quota underfill/overfill ({accumulated_tokens:,} vs target {target_quota:,}, ratio={diff_ratio:.4f} > 0.0005).")

            record_counts[source_name] = accumulated_docs
            token_counts[source_name] = accumulated_tokens

        shards_a = self.layer_a_out_io.write_records_to_shards(canonical_selected_records, shard_prefix="part")
        shards_b = self.layer_b_out_io.write_records_to_shards(tokenized_selected_records, shard_prefix="part")

        print(f"Stage 09 Tokenization & Quota Selection complete: Selected {len(canonical_selected_records):,} documents.")

        return {
            "record_counts": record_counts,
            "token_counts": token_counts,
            "output_hashes": {
                "layer_a_shards": len(shards_a),
                "layer_b_shards": len(shards_b),
                "tokenizer_sha256": tok_sha256
            }
        }
