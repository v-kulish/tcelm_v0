import pytest
from src.tcelm_corpus.stages.s05_dedup import get_20grams, compute_minhash, jaccard_sim

def test_lsh_minhash_20grams():
    text1 = "The quick brown fox jumps over the lazy dog " * 5
    text2 = "The quick brown fox jumps over the lazy dog " * 5
    text3 = "Completely different text content about quantum physics and astrophysics."

    ng1 = get_20grams(text1)
    ng2 = get_20grams(text2)
    ng3 = get_20grams(text3)

    assert len(ng1) > 0
    assert jaccard_sim(ng1, ng2) == 1.0
    assert jaccard_sim(ng1, ng3) == 0.0

    mh1 = compute_minhash(ng1, num_perm=128)
    mh2 = compute_minhash(ng2, num_perm=128)
    assert mh1 == mh2
