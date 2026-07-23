import hashlib
import struct
from dataclasses import dataclass
from typing import Iterator, Dict, Any, List, Optional
from datasets import load_dataset

def compute_deterministic_priority(corpus_version: str, source: str, doc_id: str, seed: int = 42) -> int:
    """
    Computes q(d) = uint64[ BLAKE3 / SHA256 ( corpus version, source, doc_id, seed ) ]
    """
    key = f"{corpus_version}:{source}:{doc_id}:{seed}".encode('utf-8')
    digest = hashlib.sha256(key).digest()[:8]
    val = struct.unpack(">Q", digest)[0] # 64-bit unsigned integer
    return val

@dataclass
class RawStreamRecord:
    doc_id: str
    source: str
    text: str
    priority: int
    metadata: Dict[str, Any]

class HFStreamIngester:
    def __init__(self, corpus_version: str = "TCELM-Corpus-v0", seed: int = 42):
        self.corpus_version = corpus_version
        self.seed = seed

    def stream_source(
        self,
        hf_repo_name: str,
        split: str = "train",
        max_records: Optional[int] = None
    ) -> Iterator[RawStreamRecord]:
        """
        Streams actual raw records from Hugging Face common-pile component datasets.
        """
        try:
            ds = load_dataset(hf_repo_name, split=split, streaming=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load streaming HuggingFace dataset {hf_repo_name}: {e}")

        count = 0
        for i, item in enumerate(ds):
            text = item.get("text", "") or item.get("content", "") or ""
            doc_id = str(item.get("id", item.get("doc_id", f"{hf_repo_name}_{i}")))
            
            priority = compute_deterministic_priority(self.corpus_version, hf_repo_name, doc_id, self.seed)

            record = RawStreamRecord(
                doc_id=doc_id,
                source=hf_repo_name,
                text=text,
                priority=priority,
                metadata=item
            )
            yield record

            count += 1
            if max_records is not None and count >= max_records:
                break
