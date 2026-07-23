import os
import json
from typing import Dict, Any, List
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

        tokenized_docs = []
        for rec in all_input:
            doc_id = rec.get("document_id") or rec.get("doc_id")
            parent_id = rec.get("parent_document_id") or doc_id

            tok_ids = json.loads(rec["token_ids_json"])
            para_spans = json.loads(rec.get("paragraph_spans_json", "[]"))
            sent_spans = json.loads(rec.get("sentence_spans_json", "[]"))
            turn_spans = json.loads(rec.get("turn_spans_json", "[]"))
            eq_spans = json.loads(rec.get("equation_spans_json", "[]"))

            tdoc = TokenizedDocument(
                document_id=doc_id,
                parent_document_id=parent_id,
                source=rec["source"],
                split=rec["split"],
                token_ids=tok_ids,
                sentence_token_spans=sent_spans,
                paragraph_token_spans=para_spans,
                turn_token_spans=turn_spans,
                equation_token_spans=eq_spans
            )
            tokenized_docs.append(tdoc)

        print("Generating Layer C Causal Packing views...")
        causal_views = self.view_generator.generate_causal_packing_views(tokenized_docs)

        print("Generating Layer C Prefix-Suffix views...")
        prefix_suffix_views = self.view_generator.generate_prefix_suffix_views(tokenized_docs)

        print("Generating Layer C Bridge Masked Span views...")
        bridge_views = self.view_generator.generate_bridge_views(tokenized_docs)

        view_records = []
        for v in causal_views + prefix_suffix_views + bridge_views:
            rec = {
                "view_id": v.view_id,
                "document_id": v.document_id,
                "view_type": v.view_type,
                "horizon": v.horizon,
                "relation": v.relation,
                "sampling_seed": v.sampling_seed,
                "input_token_ids_json": json.dumps(v.input_token_ids),
                "target_token_ids_json": json.dumps(v.target_token_ids),
                "loss_mask_json": json.dumps(getattr(v, "loss_mask", [1] * len(v.target_token_ids))),
                "input_token_count": len(v.input_token_ids),
                "target_token_count": len(v.target_token_ids),
                "metadata_json": json.dumps(v.metadata)
            }
            view_records.append(rec)

        written_shards = self.shard_io.write_records_to_shards(view_records, shard_prefix="part")

        return {
            "record_counts": {
                "causal_views": len(causal_views),
                "prefix_suffix_views": len(prefix_suffix_views),
                "bridge_views": len(bridge_views)
            },
            "output_hashes": {"shard_count": len(written_shards)}
        }
