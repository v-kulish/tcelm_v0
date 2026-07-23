# TCELM-Corpus-v0 Data Processing Pipeline

This repository contains the complete, modular data processing pipeline for building the **TCELM-Corpus-v0** dataset from the 18 official filtered Common Pile v0.1 component repositories on Hugging Face.

---

## 1. Quick Start

### Installation
```bash
pip install -e .
```

### Running the Pipeline
To run a small-scale real data verification with a record limit per source:
```bash
python3 run_pipeline.py --scale 50M --output-dir output_run_50m --max-records-per-source 20
```

To run a full quota-driven pipeline (e.g. 50M, 1B, 3B, or 5B scale target):
```bash
python3 run_pipeline.py --config config/corpus_v0_3b.json --scale 3B --output-dir output_run_3b
```

---

## 2. Configuration & Quotas

The source quotas and pipeline parameters are specified in [`config/corpus_v0_3b.json`](--/config/corpus_v0_3b.json):

- **18 Component Repositories**:
  - `common-pile/cccc_filtered` (30.00%)
  - `common-pile/wikimedia_filtered` (15.00%)
  - `common-pile/stackexchange_filtered` (10.00%)
  - `common-pile/doab_filtered` (7.00%)
  - `common-pile/project_gutenberg_filtered` (4.00%)
  - `common-pile/pre_1929_books_filtered` (3.00%)
  - `common-pile/pressbooks_filtered` (2.20%)
  - `common-pile/arxiv_papers_filtered` (6.00%)
  - `common-pile/peS2o_filtered` (7.00%)
  - `common-pile/pubmed_filtered` (4.00%)
  - `common-pile/libretexts_filtered` (1.60%)
  - `common-pile/oercommons_filtered` (0.20%)
  - `common-pile/uk_hansard_filtered` (2.00%)
  - `common-pile/usgpo_filtered` (1.00%)
  - `common-pile/regulations_filtered` (1.00%)
  - `common-pile/caselaw_access_project_filtered` (1.00%)
  - `common-pile/github_archive_filtered` (4.95%)
  - `common-pile/python_enhancement_proposals_filtered` (0.05%)

- **Oversampling**: Deterministic BLAKE3 priority hashing $q(d) = \text{uint64}(\text{BLAKE3}(\text{corpus\_version}, \text{source}, \text{doc\_id}, \text{seed}))$, retaining 1.35x initial quota pool.

---

## 3. Data Layers

- **Layer A**: Canonical Parquet records with Unicode NFC, PII redaction (`<EMAIL>`, `<PHONE>`, `<IP>`, `<API_KEY>`, `<PRIVATE_KEY>`), structural spans (`sentence_spans`, `paragraph_spans`, `turn_spans`, `equation_spans`), quality scores, and segment adjacency tracking.
- **Layer B**: Tokenized Parquet records with 32,768 byte-level BPE tokenizer and structural token offsets.
- **Layer C**: Derived experiment views (Causal 4k sequence packing with `<EOS><DOC>` markers, Prefix-to-Suffix trajectory views, Bridge masked span views).

---

## 4. Audit Reports

The pipeline automatically generates 7 mandatory audit reports under `<output_dir>/reports/`:
1. `01_source_report.md`
2. `02_quality_report.md`
3. `03_deduplication_report.md`
4. `04_structure_report.md`
5. `05_tokenizer_report.md`
6. `06_split_report.md`
7. `07_benchmark_report.md`

---

## 5. Testing

Run all unit and integration tests using `pytest`:
```bash
pytest tests/ -v
```
