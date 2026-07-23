import os
import json
from typing import Dict, List, Any
from .schema import CanonicalDocument, TokenizedDocument

class MandatoryReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_all_reports(
        self,
        canonical_docs: List[CanonicalDocument],
        tokenized_docs: List[TokenizedDocument],
        rejection_counts: Dict[str, int],
        contamination_logs: List[Dict[str, Any]],
        dedup_stats: Dict[str, Any]
    ) -> Dict[str, str]:
        report_files = {}

        # 1. Source Report
        source_report = self._build_source_report(canonical_docs, tokenized_docs)
        s_file = os.path.join(self.output_dir, "01_source_report.md")
        with open(s_file, "w", encoding="utf-8") as f:
            f.write(source_report)
        report_files["source_report"] = s_file

        # 2. Quality Report
        quality_report = self._build_quality_report(canonical_docs, rejection_counts)
        q_file = os.path.join(self.output_dir, "02_quality_report.md")
        with open(q_file, "w", encoding="utf-8") as f:
            f.write(quality_report)
        report_files["quality_report"] = q_file

        # 3. Deduplication Report
        dedup_report = self._build_dedup_report(dedup_stats)
        d_file = os.path.join(self.output_dir, "03_deduplication_report.md")
        with open(d_file, "w", encoding="utf-8") as f:
            f.write(dedup_report)
        report_files["dedup_report"] = d_file

        # 4. Structure Report
        struct_report = self._build_structure_report(canonical_docs)
        st_file = os.path.join(self.output_dir, "04_structure_report.md")
        with open(st_file, "w", encoding="utf-8") as f:
            f.write(struct_report)
        report_files["structure_report"] = st_file

        # 5. Tokenizer Report
        tok_report = self._build_tokenizer_report(tokenized_docs)
        t_file = os.path.join(self.output_dir, "05_tokenizer_report.md")
        with open(t_file, "w", encoding="utf-8") as f:
            f.write(tok_report)
        report_files["tokenizer_report"] = t_file

        # 6. Split Report
        split_report = self._build_split_report(canonical_docs, tokenized_docs)
        sp_file = os.path.join(self.output_dir, "06_split_report.md")
        with open(sp_file, "w", encoding="utf-8") as f:
            f.write(split_report)
        report_files["split_report"] = sp_file

        # 7. Benchmark Report
        bm_report = self._build_benchmark_report(contamination_logs)
        b_file = os.path.join(self.output_dir, "07_benchmark_report.md")
        with open(b_file, "w", encoding="utf-8") as f:
            f.write(bm_report)
        report_files["benchmark_report"] = b_file

        return report_files

    def _build_source_report(self, cdocs: List[CanonicalDocument], tdocs: List[TokenizedDocument]) -> str:
        doc_count_by_source = {}
        for d in cdocs:
            doc_count_by_source[d.source] = doc_count_by_source.get(d.source, 0) + 1

        tokens_by_source = {}
        for td in tdocs:
            tokens_by_source[td.source] = tokens_by_source.get(td.source, 0) + len(td.token_ids)

        lines = [
            "# 1. Source Quota and Ingestion Audit Report",
            "",
            "| Source Repository | Document Count | Total Retained Tokens | Share |",
            "| --- | --- | --- | --- |"
        ]
        tot_tok = sum(tokens_by_source.values()) or 1
        for src, count in doc_count_by_source.items():
            toks = tokens_by_source.get(src, 0)
            share = (toks / tot_tok) * 100
            lines.append(f"| `{src}` | {count:,} | {toks:,} | {share:.2f}% |")

        return "\n".join(lines)

    def _build_quality_report(self, cdocs: List[CanonicalDocument], rejections: Dict[str, int]) -> str:
        lines = [
            "# 2. Quality and Normalization Audit Report",
            "",
            "## Rejections by Rule",
            "",
            "| Rule Triggered | Count |",
            "| --- | --- |"
        ]
        for rule, count in rejections.items():
            lines.append(f"| `{rule}` | {count:,} |")

        tot_pii = sum(d.quality.pii_count for d in cdocs)
        lines.extend([
            "",
            f"**Total Redacted PII Instances**: {tot_pii:,}",
            f"**Total Retained Canonical Segments**: {len(cdocs):,}"
        ])
        return "\n".join(lines)

    def _build_dedup_report(self, stats: Dict[str, Any]) -> str:
        lines = [
            "# 3. Deduplication Audit Report",
            "",
            f"- **Exact Document Duplicates Removed**: {stats.get('exact_duplicates_removed', 0):,}",
            f"- **Frequent Paragraphs Stripped**: {stats.get('frequent_paragraphs_stripped', 0):,}",
            f"- **Fuzzy Duplicate Documents Clusters**: {stats.get('fuzzy_duplicate_clusters', 0):,}",
            f"- **Final Retained Unique Documents**: {stats.get('final_retained_docs', 0):,}"
        ]
        return "\n".join(lines)

    def _build_structure_report(self, cdocs: List[CanonicalDocument]) -> str:
        tot_paras = sum(len(d.structure.paragraph_spans) for d in cdocs)
        tot_sents = sum(len(d.structure.sentence_spans) for d in cdocs)
        tot_turns = sum(len(d.structure.turn_spans) for d in cdocs)
        tot_eqs = sum(len(d.structure.equation_spans) for d in cdocs)

        lines = [
            "# 4. Structural Segmentation Audit Report",
            "",
            f"- **Total Paragraph Spans**: {tot_paras:,}",
            f"- **Total Sentence Spans**: {tot_sents:,}",
            f"- **Total Structural Turn Spans**: {tot_turns:,}",
            f"- **Total Equation Spans**: {tot_eqs:,}"
        ]
        return "\n".join(lines)

    def _build_tokenizer_report(self, tdocs: List[TokenizedDocument]) -> str:
        tot_tokens = sum(len(td.token_ids) for td in tdocs)
        lines = [
            "# 5. Tokenizer Audit Report",
            "",
            f"- **Vocabulary Size**: 32,768",
            f"- **Total Corpus Tokens**: {tot_tokens:,}",
            f"- **Mean Tokens per Document Segment**: {(tot_tokens/max(len(tdocs), 1)):.1f}"
        ]
        return "\n".join(lines)

    def _build_split_report(self, cdocs: List[CanonicalDocument], tdocs: List[TokenizedDocument]) -> str:
        split_counts = {}
        for td in tdocs:
            split_counts[td.split] = split_counts.get(td.split, 0) + len(td.token_ids)

        tot_tok = sum(split_counts.values()) or 1
        lines = [
            "# 6. Data Split Audit Report",
            "",
            "| Split | Token Count | Share | Target Share |",
            "| --- | --- | --- | --- |"
        ]
        targets = {"train": "99.70%", "validation": "0.10%", "test": "0.10%", "trajectory_holdout": "0.10%"}
        for sp, count in split_counts.items():
            share = (count / tot_tok) * 100
            lines.append(f"| `{sp}` | {count:,} | {share:.2f}% | {targets.get(sp, 'N/A')} |")

        return "\n".join(lines)

    def _build_benchmark_report(self, contamination_logs: List[Dict[str, Any]]) -> str:
        lines = [
            "# 7. Benchmark Decontamination Audit Report",
            "",
            f"- **Contaminated Benchmark Match Logs**: {len(contamination_logs):,}",
            "",
            "| Benchmark | Item ID | Matched Doc ID | Match Type | Action |",
            "| --- | --- | --- | --- | --- |"
        ]
        for log in contamination_logs[:50]: # show first 50
            lines.append(f"| `{log.get('benchmark_name')}` | `{log.get('item_id')}` | `{log.get('matched_document_id')}` | `{log.get('match_type')}` | `{log.get('action')}` |")

        return "\n".join(lines)
