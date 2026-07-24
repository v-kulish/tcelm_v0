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
from ..views import DerivedViewGenerator
from ..schema import TokenizedDocument

class Stage11GenerateViews(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("11_generate_views", output_dir, config)
        self.output_dir = output_dir
        stage_09_b_dir = os.path.join(output_dir, "stages", "09_tokenize_select", "layer_b_selected")
        self.input_io = ParquetShardIO(stage_09_b_dir)
        self.view_generator = DerivedViewGenerator(seed=self.config.seed)
        self.tokenizer = BPECorpusTokenizer(
            vocab_size=self.config.tokenizer.vocab_size,
            special_tokens=self.config.tokenizer.special_tokens
        )

    def _compute_current_layer_b_digest(self) -> str:
        layer_b_dir = os.path.join(self.output_dir, "stages", "09_tokenize_select", "layer_b_selected")
        if not os.path.exists(layer_b_dir):
            raise RuntimeError(f"Freeze Verification Failure: Layer B directory missing at `{layer_b_dir}` in Stage 11 cache key calculation.")
        
        shard_files = sorted([f for f in os.listdir(layer_b_dir) if f.endswith(".parquet")])
        if not shard_files:
            raise RuntimeError(f"Freeze Verification Failure: 0 Layer B Parquet shards found in `{layer_b_dir}` in Stage 11 cache key calculation.")
        
        items = []
        for sf in shard_files:
            sp = os.path.join(layer_b_dir, sf)
            sz = os.path.getsize(sp)
            h = StageManifest.compute_file_hash(sp)
            items.append((sf, sz, h))

        return hashlib.sha256(json.dumps(items, sort_keys=True).encode("utf-8")).hexdigest()

    def get_additional_cache_inputs(self) -> Dict[str, str]:
        tok_path = os.path.join(self.output_dir, "stages", "08_train_tokenizer", "tokenizer.json")
        freeze_man_path = os.path.join(self.output_dir, "stages", "10_freeze", "freeze_manifest.json")

        if not os.path.exists(tok_path):
            raise RuntimeError(f"Freeze Verification Failure: Tokenizer missing at `{tok_path}` in Stage 11 cache key calculation.")

        if not os.path.exists(freeze_man_path):
            raise RuntimeError(f"Freeze Verification Failure: Stage 10 freeze_manifest.json missing at `{freeze_man_path}` in Stage 11 cache key calculation.")

        tok_sha = StageManifest.compute_file_hash(tok_path)
        freeze_sha = StageManifest.compute_file_hash(freeze_man_path)
        layer_b_digest = self._compute_current_layer_b_digest()

        return {
            "tokenizer_sha256": tok_sha,
            "freeze_manifest_sha256": freeze_sha,
            "current_layer_b_artifact_digest": layer_b_digest
        }

    def run_stage(self) -> Dict[str, Any]:
        tok_path = os.path.join(self.output_dir, "stages", "08_train_tokenizer", "tokenizer.json")
        freeze_man_path = os.path.join(self.output_dir, "stages", "10_freeze", "freeze_manifest.json")

        if not os.path.exists(tok_path):
            raise RuntimeError(f"Tokenizer file missing at `{tok_path}` in Stage 11 GenerateViews.")

        if not os.path.exists(freeze_man_path):
            raise RuntimeError(f"Freeze Verification Failure: Required freeze manifest missing at `{freeze_man_path}` in Stage 11 GenerateViews.")

        tok_sha256 = StageManifest.compute_file_hash(tok_path)

        with open(freeze_man_path, "r", encoding="utf-8") as f:
            freeze_man = json.load(f)

        frozen_tok_sha = freeze_man.get("tokenizer_sha256")
        if not frozen_tok_sha or tok_sha256 != frozen_tok_sha:
            raise RuntimeError(f"Tokenizer Mismatch: Active Stage 11 tokenizer hash ({tok_sha256[:10]}...) does not match Stage 10 freeze_manifest.json ({str(frozen_tok_sha)[:10]}...).")

        # Layer B Shard Hash & Path Verification against Stage 10 freeze_manifest.json
        frozen_shards_info = freeze_man.get("layer_b_shards", {})
        layer_b_dir = os.path.join(self.output_dir, "stages", "09_tokenize_select", "layer_b_selected")
        current_shard_files = sorted([f for f in os.listdir(layer_b_dir) if f.endswith(".parquet")])
        current_shard_rel_paths = set(current_shard_files)

        expected_shard_rel_paths = set(os.path.basename(p) for p in frozen_shards_info.keys())

        if current_shard_rel_paths != expected_shard_rel_paths:
            raise RuntimeError(f"Layer B Shard Set Mismatch: Current Layer B shards ({sorted(current_shard_rel_paths)}) do not match Stage 10 freeze manifest ({sorted(expected_shard_rel_paths)}).")

        for sf in current_shard_files:
            sp = os.path.join(layer_b_dir, sf)
            current_hash = StageManifest.compute_file_hash(sp)
            current_size = os.path.getsize(sp)
            
            matching_key = next((k for k in frozen_shards_info.keys() if os.path.basename(k) == sf), None)
            if not matching_key:
                raise RuntimeError(f"Freeze Verification Failure: Shard `{sf}` missing from freeze manifest.")

            expected_info = frozen_shards_info[matching_key]
            if isinstance(expected_info, str):
                expected_hash = expected_info
                expected_size = None
            else:
                expected_hash = expected_info.get("sha256")
                expected_size = expected_info.get("size_bytes")

            if current_hash != expected_hash:
                raise RuntimeError(f"Layer B Shard Hash Violation: Shard `{sf}` SHA-256 ({current_hash}) does not match frozen checksum ({expected_hash}). Data modified after freeze!")

            if expected_size is not None and current_size != expected_size:
                raise RuntimeError(f"Layer B Shard Size Violation: Shard `{sf}` byte size ({current_size}) does not match frozen byte size ({expected_size}).")

        self.tokenizer.load_tokenizer(tok_path)

        # Strict Special Token Resolution
        bos_id = self.tokenizer.tokenizer.token_to_id("<BOS>")
        eos_id = self.tokenizer.tokenizer.token_to_id("<EOS>")
        doc_id = self.tokenizer.tokenizer.token_to_id("<DOC>")
        mask_span_id = self.tokenizer.tokenizer.token_to_id("<MASK_SPAN>")

        if any(t is None for t in [bos_id, eos_id, doc_id, mask_span_id]):
            raise RuntimeError(f"Special Token Resolution Failure: One or more special tokens (<BOS>={bos_id}, <EOS>={eos_id}, <DOC>={doc_id}, <MASK_SPAN>={mask_span_id}) could not be resolved from tokenizer.")
        
        if len({bos_id, eos_id, doc_id, mask_span_id}) != 4:
            raise RuntimeError(f"Special Token Collision: Special tokens (<BOS>, <EOS>, <DOC>, <MASK_SPAN>) must resolve to 4 distinct IDs, got {[bos_id, eos_id, doc_id, mask_span_id]}.")

        all_input = list(self.input_io.read_shards())
        if not all_input:
            raise RuntimeError("Stage '11_generate_views' received 0 input records from Stage 09 Layer B.")

        split_docs = defaultdict(list)
        doc_split_map = {}

        for rec in tqdm(all_input, desc="Loading Tokenized Records for Views", unit="doc"):
            doc_id_val = rec.get("document_id") or rec.get("doc_id")
            parent_id = rec.get("parent_document_id") or doc_id_val
            split = rec.get("split", "train")

            tok_ids = json.loads(rec["token_ids_json"])
            para_spans = json.loads(rec.get("paragraph_spans_json", "[]"))
            sent_spans = json.loads(rec.get("sentence_spans_json", "[]"))
            turn_spans = json.loads(rec.get("turn_spans_json", "[]"))
            eq_spans = json.loads(rec.get("equation_spans_json", "[]"))

            tdoc = TokenizedDocument(
                document_id=doc_id_val,
                parent_document_id=parent_id,
                source=rec["source"],
                split=split,
                token_ids=tok_ids,
                sentence_token_spans=sent_spans,
                paragraph_token_spans=para_spans,
                turn_token_spans=turn_spans,
                equation_token_spans=eq_spans
            )
            split_docs[split].append(tdoc)
            doc_split_map[doc_id_val] = split

        all_splits = ["train", "validation", "test", "trajectory_holdout"]
        total_view_counts = defaultdict(int)
        split_shard_counts = {}

        print("Stage 11 View Generation: Partitioning documents before view generation to guarantee zero cross-split leakage...")

        for split_name in all_splits:
            docs_in_split = split_docs.get(split_name, [])
            if not docs_in_split:
                continue

            allow_packing = (split_name == "train")
            causal_views = self.view_generator.generate_causal_packing_views(
                docs_in_split,
                split=split_name,
                allow_packing=allow_packing,
                bos_id=bos_id,
                eos_id=eos_id,
                doc_id=doc_id
            )
            prefix_suffix_views = self.view_generator.generate_prefix_suffix_views(docs_in_split, split=split_name)
            bridge_views = self.view_generator.generate_bridge_views(docs_in_split, split=split_name, mask_span_id=mask_span_id)

            views_for_split = causal_views + prefix_suffix_views + bridge_views

            # Assert zero cross-split leakage invariant
            for v in views_for_split:
                for s_doc_id in v.source_document_ids:
                    if doc_split_map[s_doc_id] != split_name:
                        raise RuntimeError(f"Split Isolation Violation: View `{v.view_id}` assigned to split `{split_name}` references document `{s_doc_id}` from split `{doc_split_map[s_doc_id]}`.")

            view_records = []
            for v in views_for_split:
                rec = {
                    "view_id": v.view_id,
                    "split": v.split,
                    "usage": v.usage,
                    "view_type": v.view_type,
                    "source_document_ids_json": json.dumps(v.source_document_ids),
                    "source_parent_document_ids_json": json.dumps(v.source_parent_document_ids),
                    "horizon": v.horizon,
                    "relation": v.relation,
                    "sampling_seed": v.sampling_seed,
                    "input_token_ids_json": json.dumps(v.input_token_ids),
                    "target_token_ids_json": json.dumps(v.target_token_ids),
                    "loss_mask_json": json.dumps(v.loss_mask),
                    "attention_mask_json": json.dumps(v.attention_mask),
                    "input_token_count": len(v.input_token_ids),
                    "target_token_count": len(v.target_token_ids),
                    "attention_mask_count": len(v.attention_mask),
                    "metadata_json": json.dumps(v.metadata)
                }
                view_records.append(rec)

            split_output_dir = os.path.join(self.stage_dir, split_name)
            split_io = ParquetShardIO(split_output_dir)
            written_shards = split_io.write_records_to_shards(view_records, shard_prefix="part")
            split_shard_counts[split_name] = len(written_shards)

            total_view_counts[f"{split_name}_causal"] += len(causal_views)
            total_view_counts[f"{split_name}_prefix_suffix"] += len(prefix_suffix_views)
            total_view_counts[f"{split_name}_bridge"] += len(bridge_views)
            total_view_counts[f"{split_name}_total"] += len(views_for_split)

            print(f"Split `{split_name}` views complete: Generated {len(views_for_split):,} view recipes ({len(causal_views):,} causal, {len(prefix_suffix_views):,} prefix-suffix, {len(bridge_views):,} bridge) in `{split_output_dir}`.")

        return {
            "record_counts": dict(total_view_counts),
            "output_hashes": {
                "split_shards": split_shard_counts,
                "tokenizer_sha256": tok_sha256,
                "special_token_ids": {
                    "<BOS>": bos_id,
                    "<EOS>": eos_id,
                    "<DOC>": doc_id,
                    "<MASK_SPAN>": mask_span_id
                }
            }
        }
