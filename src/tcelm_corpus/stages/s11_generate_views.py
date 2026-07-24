import os
import json
from typing import Dict, Any, List
from collections import defaultdict
from tqdm import tqdm
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO
from ..views import DerivedViewGenerator
from ..schema import TokenizedDocument

class Stage11GenerateViews(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("11_generate_views", output_dir, config)
        stage_09_b_dir = os.path.join(output_dir, "stages", "09_tokenize_select", "layer_b_selected")
        self.input_io = ParquetShardIO(stage_09_b_dir)
        self.view_generator = DerivedViewGenerator(seed=self.config.seed)

    def run_stage(self) -> Dict[str, Any]:
        all_input = list(self.input_io.read_shards())
        if not all_input:
            raise RuntimeError("Stage '11_generate_views' received 0 input records from Stage 09 Layer B.")

        split_docs = defaultdict(list)
        doc_split_map = {}

        for rec in tqdm(all_input, desc="Loading Tokenized Records for Views", unit="doc"):
            doc_id = rec.get("document_id") or rec.get("doc_id")
            parent_id = rec.get("parent_document_id") or doc_id
            split = rec.get("split", "train")

            tok_ids = json.loads(rec["token_ids_json"])
            para_spans = json.loads(rec.get("paragraph_spans_json", "[]"))
            sent_spans = json.loads(rec.get("sentence_spans_json", "[]"))
            turn_spans = json.loads(rec.get("turn_spans_json", "[]"))
            eq_spans = json.loads(rec.get("equation_spans_json", "[]"))

            tdoc = TokenizedDocument(
                document_id=doc_id,
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
            doc_split_map[doc_id] = split

        all_splits = ["train", "validation", "test", "trajectory_holdout"]
        total_view_counts = defaultdict(int)
        split_shard_counts = {}

        print("Stage 11 View Generation: Partitioning documents before view generation to guarantee zero cross-split leakage...")

        for split_name in all_splits:
            docs_in_split = split_docs.get(split_name, [])
            if not docs_in_split:
                continue

            allow_packing = (split_name == "train")
            causal_views = self.view_generator.generate_causal_packing_views(docs_in_split, split=split_name, allow_packing=allow_packing)
            prefix_suffix_views = self.view_generator.generate_prefix_suffix_views(docs_in_split, split=split_name)
            bridge_views = self.view_generator.generate_bridge_views(docs_in_split, split=split_name)

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
                    "input_token_count": len(v.input_token_ids),
                    "target_token_count": len(v.target_token_ids),
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
            "output_hashes": split_shard_counts
        }
