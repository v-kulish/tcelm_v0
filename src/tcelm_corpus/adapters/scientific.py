from typing import Dict, Any, Tuple
from .base import BaseSourceAdapter

class ScientificAdapter(BaseSourceAdapter):
    def extract_record(self, raw_record: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        license_status = raw_record.get("license", raw_record.get("licence", "")) or "missing"
        text = raw_record.get("text", "") or raw_record.get("abstract", "") or raw_record.get("content", "")

        field_family = raw_record.get("field_family", raw_record.get("field", "cs_engineering"))
        title = raw_record.get("title", "")

        metadata = {
            "title": title,
            "url": raw_record.get("url", raw_record.get("arxiv_id", "")),
            "field_family": field_family,
            "license": license_status
        }

        return text, license_status, metadata
