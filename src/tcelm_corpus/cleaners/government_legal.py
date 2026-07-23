import re
from typing import List, Dict, Any, Optional
from .generic import CleaningResult

class GovernmentLegalCleaner:
    def clean_hansard(self, session_title: str, utterances: List[Dict[str, str]]) -> CleaningResult:
        if not utterances:
            return CleaningResult("", True, "empty_hansard_utterances", {})

        formatted_parts = [f"<SESSION>\n{session_title.strip()}\n</SESSION>"]
        speech_count = 0

        for u in utterances:
            speaker = u.get("speaker", "SPEAKER").strip()
            text = u.get("text", "").strip()
            if not text or len(text.split()) < 4:
                continue

            formatted_parts.append(f"<SPEAKER>{speaker}</SPEAKER>\n<UTTERANCE>\n{text}\n</UTTERANCE>")
            speech_count += 1

        if speech_count == 0:
            return CleaningResult("", True, "no_hansard_speeches_retained", {})

        full_text = "\n\n".join(formatted_parts)
        words = full_text.split()

        if len(words) < 256:
            return CleaningResult(full_text, True, f"hansard_below_256_{len(words)}", {})

        return CleaningResult(full_text, False, None, {"approx_tokens": len(words), "speech_count": speech_count})

    def clean_usgpo_regulations_caselaw(self, text: str, source_type: str = "usgpo") -> CleaningResult:
        if not text:
            return CleaningResult("", True, "empty_text", {})

        # Remove repeated docket metadata / citation tables
        cleaned = re.sub(r'(?:Docket No\.|FOR FURTHER INFORMATION CONTACT:|Page \d+ of \d+).*?\n', '', text, flags=re.I)
        words = cleaned.split()

        if len(words) < 256:
            return CleaningResult(cleaned, True, f"{source_type}_below_256_{len(words)}", {})

        return CleaningResult(cleaned, False, None, {"approx_tokens": len(words)})
