import re
from typing import Optional
from .generic import CleaningResult

class EducationalCleaner:
    def clean_libretexts(self, text: str) -> CleaningResult:
        if not text:
            return CleaningResult("", True, "empty_text", {})

        # Strip navigation, PDF download buttons, donation prompts
        cleaned = re.sub(r'(?:Download PDF|Donate to LibreTexts|Back to top|Edit page).*?\n', '', text, flags=re.I)
        words = cleaned.split()

        if len(words) < 512:
            return CleaningResult(cleaned, True, f"libretexts_below_512_{len(words)}", {})

        return CleaningResult(cleaned, False, None, {"approx_tokens": len(words)})

    def clean_oercommons(self, text: str) -> CleaningResult:
        if not text:
            return CleaningResult("", True, "empty_text", {})

        # Reject catalog entries without full lesson content
        if "Catalog Record" in text or "Resource Description:" in text:
            if len(text.split()) < 300:
                return CleaningResult(text, True, "oercommons_catalog_stub_rejected", {})

        words = text.split()
        if len(words) < 256:
            return CleaningResult(text, True, f"oercommons_below_256_{len(words)}", {})

        return CleaningResult(text, False, None, {"approx_tokens": len(words)})
