import os
from typing import List, Dict, Tuple, Optional, Any
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders
from .schema import CanonicalDocument, TokenizedDocument

DEFAULT_SPECIAL_TOKENS = [
    "<BOS>",
    "<EOS>",
    "<DOC>",
    "<SECTION>",
    "<PARAGRAPH>",
    "<TURN>",
    "<QUESTION>",
    "<ANSWER>",
    "<COMMENT>",
    "<EQUATION>",
    "<CODE>",
    "<MASK_SPAN>",
    "<EMAIL>",
    "<PHONE>",
    "<IP>",
    "<API_KEY>",
    "<PRIVATE_KEY>"
]

class BPECorpusTokenizer:
    def __init__(self, vocab_size: int = 32768, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS
        self.tokenizer: Optional[Tokenizer] = None

    def train_from_texts(self, texts: List[str], save_path: Optional[str] = None) -> Tokenizer:
        # Initialize Byte-Level BPE model
        tokenizer = Tokenizer(models.BPE(unk_token="<EOS>"))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)
        self.tokenizer = tokenizer

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            tokenizer.save(save_path)

        return tokenizer

    def load_tokenizer(self, model_path: str):
        self.tokenizer = Tokenizer.from_file(model_path)

    def encode_document(self, doc: CanonicalDocument) -> TokenizedDocument:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded.")

        encoded = self.tokenizer.encode(doc.normalized_text)
        token_ids = encoded.ids
        offsets = encoded.offsets # List[Tuple[int, int]]

        # Map character spans in doc.structure to token span indices
        def char_spans_to_token_spans(char_spans: List[List[int]]) -> List[List[int]]:
            tok_spans = []
            for c_start, c_end in char_spans:
                t_start = None
                t_end = None
                for idx, (o_start, o_end) in enumerate(offsets):
                    if t_start is None and o_start >= c_start:
                        t_start = idx
                    if o_end <= c_end:
                        t_end = idx + 1
                if t_start is not None and t_end is not None and t_end > t_start:
                    tok_spans.append([t_start, t_end])
            return tok_spans

        sent_token_spans = char_spans_to_token_spans(doc.structure.sentence_spans)
        para_token_spans = char_spans_to_token_spans(doc.structure.paragraph_spans)
        turn_token_spans = char_spans_to_token_spans(doc.structure.turn_spans)
        eq_token_spans = char_spans_to_token_spans(doc.structure.equation_spans)

        return TokenizedDocument(
            document_id=doc.document_id,
            parent_document_id=doc.parent_document_id,
            source=doc.source,
            split=doc.split,
            token_ids=token_ids,
            sentence_token_spans=sent_token_spans,
            paragraph_token_spans=para_token_spans,
            section_token_spans=[],
            turn_token_spans=turn_token_spans,
            equation_token_spans=eq_token_spans
        )
