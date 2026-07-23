import random
from typing import List, Dict, Any, Tuple, Optional
from .schema import TokenizedDocument, LayerCViewRecord

class DerivedViewGenerator:
    def __init__(self, ctx_length: int = 4096, seed: int = 42):
        self.ctx_length = ctx_length
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_causal_packing_views(
        self,
        tokenized_docs: List[TokenizedDocument],
        bos_id: int = 0,
        eos_id: int = 1,
        doc_id: int = 2
    ) -> List[LayerCViewRecord]:
        views: List[LayerCViewRecord] = []
        if not tokenized_docs:
            return views

        view_counter = 0
        current_seq: List[int] = [bos_id]
        doc_ids_in_seq: List[str] = []

        for doc in tokenized_docs:
            tokens = doc.token_ids
            if not tokens:
                continue

            # 80% contiguous document sampling logic
            if len(tokens) >= self.ctx_length:
                # Pick starting offset: 60% paragraph boundary, 25% section, 15% random
                r = self.rng.random()
                if r < 0.60 and doc.paragraph_token_spans:
                    start_idx = self.rng.choice(doc.paragraph_token_spans)[0]
                else:
                    start_idx = self.rng.randint(0, max(0, len(tokens) - self.ctx_length))

                sub_tokens = tokens[start_idx:start_idx + self.ctx_length]
                if len(sub_tokens) == self.ctx_length:
                    v_record = LayerCViewRecord(
                        view_id=f"causal_single_{view_counter}",
                        document_id=doc.document_id,
                        view_type="causal_single_doc",
                        input_token_ids=sub_tokens[:-1],
                        target_token_ids=sub_tokens[1:],
                        horizon=1,
                        relation="causal",
                        sampling_seed=self.seed,
                        metadata={"start_idx": start_idx, "doc_id": doc.document_id}
                    )
                    views.append(v_record)
                    view_counter += 1
            else:
                # 20% document packing logic
                if len(current_seq) + len(tokens) + 2 <= self.ctx_length:
                    current_seq.extend(tokens)
                    current_seq.extend([eos_id, doc_id])
                    doc_ids_in_seq.append(doc.document_id)
                else:
                    # Pad or truncate current packed sequence to ctx_length
                    if len(current_seq) > 1:
                        if len(current_seq) < self.ctx_length + 1:
                            # Pad with EOS
                            current_seq.extend([eos_id] * (self.ctx_length + 1 - len(current_seq)))
                        sub_seq = current_seq[:self.ctx_length + 1]
                        v_record = LayerCViewRecord(
                            view_id=f"causal_packed_{view_counter}",
                            document_id=",".join(doc_ids_in_seq[:3]),
                            view_type="causal_packed_doc",
                            input_token_ids=sub_seq[:-1],
                            target_token_ids=sub_seq[1:],
                            horizon=1,
                            relation="causal_packed",
                            sampling_seed=self.seed,
                            metadata={"doc_count": len(doc_ids_in_seq)}
                        )
                        views.append(v_record)
                        view_counter += 1

                    current_seq = [bos_id] + tokens + [eos_id, doc_id]
                    doc_ids_in_seq = [doc.document_id]

        return views

    def generate_prefix_suffix_views(
        self,
        tokenized_docs: List[TokenizedDocument]
    ) -> List[LayerCViewRecord]:
        views: List[LayerCViewRecord] = []
        view_counter = 0

        for doc in tokenized_docs:
            tokens = doc.token_ids
            if len(tokens) < 512:
                continue

            # Prefix length options
            prefix_len = self.rng.choice([256, 512, 1024])
            if len(tokens) <= prefix_len + 64:
                continue

            prefix = tokens[:prefix_len]
            # Horizon sampling: 1, 4, 16, 64, 256 tokens
            horizon = self.rng.choice([1, 4, 16, 64, 256])
            target = tokens[prefix_len:prefix_len + horizon]

            if target:
                v_record = LayerCViewRecord(
                    view_id=f"prefix_suffix_{view_counter}",
                    document_id=doc.document_id,
                    view_type="prefix_suffix",
                    input_token_ids=prefix,
                    target_token_ids=target,
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
        mask_span_id: int = 11
    ) -> List[LayerCViewRecord]:
        views: List[LayerCViewRecord] = []
        view_counter = 0

        for doc in tokenized_docs:
            if not doc.paragraph_token_spans or len(doc.paragraph_token_spans) < 3:
                continue

            # Remove middle paragraph span as bridge
            p_idx = self.rng.randint(1, len(doc.paragraph_token_spans) - 2)
            b_start, b_end = doc.paragraph_token_spans[p_idx]

            left_ctx = doc.token_ids[:b_start]
            bridge_target = doc.token_ids[b_start:b_end]
            right_ctx = doc.token_ids[b_end:]

            input_seq = left_ctx + [mask_span_id] + right_ctx
            if len(input_seq) > self.ctx_length:
                input_seq = input_seq[:self.ctx_length]

            v_record = LayerCViewRecord(
                view_id=f"bridge_{view_counter}",
                document_id=doc.document_id,
                view_type="bridge_masked_span",
                input_token_ids=input_seq,
                target_token_ids=bridge_target,
                horizon=len(bridge_target),
                relation="bridge_recovery",
                sampling_seed=self.seed,
                metadata={"bridge_span_tokens": len(bridge_target)}
            )
            views.append(v_record)
            view_counter += 1

        return views
