import random
from typing import List, Dict, Any, Tuple, Optional
from .schema import TokenizedDocument, LayerCViewRecord

class DerivedViewGenerator:
    def __init__(self, ctx_length: int = 4096, seed: int = 42):
        self.ctx_length = ctx_length
        self.seed = seed
        self.rng = random.Random(seed)

    def finalize_packed_sequence(
        self,
        real_sequence: List[int],
        source_document_ids: List[str],
        source_parent_document_ids: List[str],
        split: str,
        view_type: str,
        view_counter: int,
        eos_id: int = 1
    ) -> LayerCViewRecord:
        C = self.ctx_length
        R = min(len(real_sequence), C + 1)

        seq = list(real_sequence[:R])
        padding_count = (C + 1) - R
        seq.extend([eos_id] * padding_count)

        input_ids = seq[:-1]
        target_ids = seq[1:]

        valid_target_count = max(0, R - 1)
        valid_input_count = min(R, C)

        loss_mask = [1] * valid_target_count + [0] * (C - valid_target_count)
        attention_mask = [1] * valid_input_count + [0] * (C - valid_input_count)

        usage_label = "pretraining" if split == "train" else "evaluation"

        assert len(input_ids) == C
        assert len(target_ids) == C
        assert len(loss_mask) == C
        assert len(attention_mask) == C
        assert sum(loss_mask) == max(0, R - 1)
        assert sum(attention_mask) == min(R, C)

        return LayerCViewRecord(
            view_id=f"{split}_{view_type}_{view_counter}",
            split=split,
            usage=usage_label,
            view_type=view_type,
            source_document_ids=list(source_document_ids),
            source_parent_document_ids=list(source_parent_document_ids),
            input_token_ids=input_ids,
            target_token_ids=target_ids,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            horizon=1,
            relation="causal" if "single" in view_type else "causal_packed",
            sampling_seed=self.seed,
            metadata={"doc_count": len(source_document_ids)}
        )

    def generate_causal_packing_views(
        self,
        tokenized_docs: List[TokenizedDocument],
        split: str = "train",
        allow_packing: bool = True,
        bos_id: int = 0,
        eos_id: int = 1,
        doc_id: int = 2
    ) -> List[LayerCViewRecord]:
        views: List[LayerCViewRecord] = []
        if not tokenized_docs:
            return views

        view_counter = 0
        current_seq: List[int] = []
        doc_ids_in_seq: List[str] = []
        parent_ids_in_seq: List[str] = []

        min_view_len = min(64, self.ctx_length)

        for doc in tokenized_docs:
            tokens = doc.token_ids
            if not tokens:
                continue

            # Documents at least context length (C-1) or packing disabled mode -> Single Document View
            if len(tokens) >= (self.ctx_length - 1) or not allow_packing:
                needed_len = self.ctx_length + 1
                if len(tokens) >= needed_len:
                    r = self.rng.random()
                    if r < 0.60 and doc.paragraph_token_spans:
                        start_idx = self.rng.choice(doc.paragraph_token_spans)[0]
                    else:
                        start_idx = self.rng.randint(0, max(0, len(tokens) - needed_len))
                    sub_tokens = tokens[start_idx:start_idx + needed_len]
                else:
                    sub_tokens = tokens

                if len(sub_tokens) >= min_view_len:
                    v_record = self.finalize_packed_sequence(
                        real_sequence=sub_tokens,
                        source_document_ids=[doc.document_id],
                        source_parent_document_ids=[doc.parent_document_id],
                        split=split,
                        view_type="causal_single_doc",
                        view_counter=view_counter,
                        eos_id=eos_id
                    )
                    views.append(v_record)
                    view_counter += 1
            else:
                # Packing logic: <BOS> doc1 <EOS> <DOC> doc2 <EOS>
                if not current_seq:
                    candidate = [bos_id] + tokens + [eos_id]
                else:
                    candidate = current_seq + [doc_id] + tokens + [eos_id]

                if len(candidate) <= self.ctx_length + 1:
                    current_seq = candidate
                    doc_ids_in_seq.append(doc.document_id)
                    parent_ids_in_seq.append(doc.parent_document_id)
                else:
                    if len(current_seq) >= 2 and doc_ids_in_seq:
                        v_record = self.finalize_packed_sequence(
                            real_sequence=current_seq,
                            source_document_ids=doc_ids_in_seq,
                            source_parent_document_ids=parent_ids_in_seq,
                            split=split,
                            view_type="causal_packed_doc",
                            view_counter=view_counter,
                            eos_id=eos_id
                        )
                        views.append(v_record)
                        view_counter += 1

                    current_seq = [bos_id] + tokens + [eos_id]
                    doc_ids_in_seq = [doc.document_id]
                    parent_ids_in_seq = [doc.parent_document_id]

        # Final buffer flush
        if len(current_seq) >= 2 and doc_ids_in_seq:
            v_record = self.finalize_packed_sequence(
                real_sequence=current_seq,
                source_document_ids=doc_ids_in_seq,
                source_parent_document_ids=parent_ids_in_seq,
                split=split,
                view_type="causal_packed_doc",
                view_counter=view_counter,
                eos_id=eos_id
            )
            views.append(v_record)
            view_counter += 1

        return views

    def generate_prefix_suffix_views(
        self,
        tokenized_docs: List[TokenizedDocument],
        split: str = "train"
    ) -> List[LayerCViewRecord]:
        views: List[LayerCViewRecord] = []
        view_counter = 0
        usage_label = "pretraining" if split == "train" else "evaluation"

        for doc in tokenized_docs:
            tokens = doc.token_ids
            if len(tokens) < 512:
                continue

            prefix_len = self.rng.choice([256, 512, 1024])
            if len(tokens) <= prefix_len + 64:
                continue

            prefix = tokens[:prefix_len]
            horizon = self.rng.choice([1, 4, 16, 64, 256])
            target = tokens[prefix_len:prefix_len + horizon]

            if target:
                v_record = LayerCViewRecord(
                    view_id=f"{split}_prefix_suffix_{view_counter}",
                    split=split,
                    usage=usage_label,
                    view_type="prefix_suffix",
                    source_document_ids=[doc.document_id],
                    source_parent_document_ids=[doc.parent_document_id],
                    input_token_ids=prefix,
                    target_token_ids=target,
                    loss_mask=[1] * len(target),
                    attention_mask=[1] * len(prefix),
                    horizon=horizon,
                    relation="trajectory_continuation",
                    sampling_seed=self.seed,
                    metadata={"prefix_len": prefix_len, "horizon": horizon}
                )
                views.append(v_record)
                view_counter += 1

        return views

    def generate_bridge_views(
        self,
        tokenized_docs: List[TokenizedDocument],
        split: str = "train",
        mask_span_id: int = 11
    ) -> List[LayerCViewRecord]:
        views: List[LayerCViewRecord] = []
        view_counter = 0
        usage_label = "pretraining" if split == "train" else "evaluation"

        for doc in tokenized_docs:
            if not doc.paragraph_token_spans or len(doc.paragraph_token_spans) < 3:
                continue

            p_idx = self.rng.randint(1, len(doc.paragraph_token_spans) - 2)
            b_start, b_end = doc.paragraph_token_spans[p_idx]

            left_ctx = doc.token_ids[:b_start]
            bridge_target = doc.token_ids[b_start:b_end]
            right_ctx = doc.token_ids[b_end:]

            input_seq = left_ctx + [mask_span_id] + right_ctx
            if len(input_seq) > self.ctx_length:
                input_seq = input_seq[:self.ctx_length]

            v_record = LayerCViewRecord(
                view_id=f"{split}_bridge_{view_counter}",
                split=split,
                usage=usage_label,
                view_type="bridge_masked_span",
                source_document_ids=[doc.document_id],
                source_parent_document_ids=[doc.parent_document_id],
                input_token_ids=input_seq,
                target_token_ids=bridge_target,
                loss_mask=[1] * len(bridge_target),
                attention_mask=[1] * len(input_seq),
                horizon=len(bridge_target),
                relation="bridge_recovery",
                sampling_seed=self.seed,
                metadata={"bridge_span_tokens": len(bridge_target)}
            )
            views.append(v_record)
            view_counter += 1

        return views
