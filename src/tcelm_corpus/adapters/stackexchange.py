from typing import Dict, Any, Tuple
from .base import BaseSourceAdapter

class StackExchangeAdapter(BaseSourceAdapter):
    def extract_record(self, raw_record: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        license_status = raw_record.get("license", raw_record.get("licence", "")) or "missing"
        
        title = (raw_record.get("title") or "").strip()
        question_body = (raw_record.get("question_body") or raw_record.get("body") or raw_record.get("text") or "").strip()
        answers = raw_record.get("answers", [])

        # Format turn structure
        parts = [
            f"<QUESTION_TITLE>\n{title}\n</QUESTION_TITLE>",
            f"<QUESTION_BODY>\n{question_body}\n</QUESTION_BODY>"
        ]

        if isinstance(answers, list):
            for ans in answers[:20]:
                if isinstance(ans, dict):
                    ans_text = ans.get("body", "").strip()
                    if ans_text:
                        parts.append(f"<ANSWER>\n{ans_text}\n</ANSWER>")

        text = "\n\n".join(parts) if question_body else raw_record.get("text", "")

        metadata = {
            "title": title,
            "url": raw_record.get("url", ""),
            "site": raw_record.get("site", "stackoverflow"),
            "license": license_status
        }

        return text, license_status, metadata
