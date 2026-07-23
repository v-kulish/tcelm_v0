from typing import Dict, Any, Tuple
from .base import BaseSourceAdapter

class BooksAdapter(BaseSourceAdapter):
    def extract_record(self, raw_record: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        license_status = raw_record.get("license", raw_record.get("licence", "")) or "missing"
        text = raw_record.get("text", "") or raw_record.get("content", "")

        metadata = {
            "title": raw_record.get("title", ""),
            "book_id": raw_record.get("book_id", raw_record.get("id", "")),
            "unigram_log_likelihood": raw_record.get("unigram_log_likelihood", -10.0),
            "license": license_status
        }
        return text, license_status, metadata

class WikimediaAdapter(BaseSourceAdapter):
    def extract_record(self, raw_record: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        license_status = raw_record.get("license", raw_record.get("licence", "")) or "missing"
        text = raw_record.get("text", "") or raw_record.get("content", "")

        metadata = {
            "title": raw_record.get("title", ""),
            "subproject": raw_record.get("subproject", raw_record.get("wiki", "wikipedia")),
            "license": license_status
        }
        return text, license_status, metadata

class CCCCAdapter(BaseSourceAdapter):
    def extract_record(self, raw_record: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        license_status = raw_record.get("license", raw_record.get("licence", "")) or "missing"
        text = raw_record.get("text", "") or raw_record.get("content", "")

        metadata = {
            "url": raw_record.get("url", raw_record.get("domain", "")),
            "title": raw_record.get("title", ""),
            "license": license_status
        }
        return text, license_status, metadata

class EducationalAdapter(BaseSourceAdapter):
    def extract_record(self, raw_record: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        license_status = raw_record.get("license", raw_record.get("licence", "")) or "missing"
        text = raw_record.get("text", "") or raw_record.get("content", "")

        metadata = {
            "title": raw_record.get("title", ""),
            "license": license_status
        }
        return text, license_status, metadata

class GovernmentLegalAdapter(BaseSourceAdapter):
    def extract_record(self, raw_record: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        license_status = raw_record.get("license", raw_record.get("licence", "")) or "missing"
        text = raw_record.get("text", "") or raw_record.get("content", "")

        utterances = raw_record.get("utterances", [])
        if "hansard" in self.source_name and isinstance(utterances, list) and utterances:
            session_title = raw_record.get("session_title", raw_record.get("title", "Hansard Debate"))
            parts = [f"<SESSION>\n{session_title}\n</SESSION>"]
            for u in utterances:
                if isinstance(u, dict):
                    spk = u.get("speaker", "SPEAKER")
                    spk_text = u.get("text", "").strip()
                    if spk_text:
                        parts.append(f"<SPEAKER>{spk}</SPEAKER>\n<UTTERANCE>\n{spk_text}\n</UTTERANCE>")
            text = "\n\n".join(parts)

        metadata = {
            "title": raw_record.get("title", ""),
            "license": license_status
        }
        return text, license_status, metadata
