import re
from typing import Optional, List
from .generic import CleaningResult

DISALLOWED_NAMESPACES_REGEX = re.compile(
    r'^(?:talk|user|wikipedia|file|mediawiki|template|help|category|portal|draft|timedtext|module|talk_page|disambiguation):',
    re.I
)

SECTION_STRIP_REGEX = re.compile(
    r'\n==+\s*(?:References|External\s+links|Further\s+reading|Bibliography|Sources|See\s+also)\s*==+.*$',
    re.DOTALL | re.I
)

class WikimediaCleaner:
    def __init__(self):
        self.allowed_projects = {"wikipedia", "wikibooks", "wikiversity", "wikivoyage", "wikinews"}

    def clean(self, text: str, title: str = "", subproject: str = "wikipedia") -> CleaningResult:
        if not text:
            return CleaningResult("", True, "empty_text", {})

        # Subproject check
        subproject_clean = subproject.lower()
        if subproject_clean not in self.allowed_projects:
            return CleaningResult(text, True, f"disallowed_wikimedia_subproject_{subproject}", {})

        # Namespace / meta page check
        if DISALLOWED_NAMESPACES_REGEX.search(title.strip()):
            return CleaningResult(text, True, f"disallowed_wikimedia_namespace_{title}", {})

        # Disambiguation or list page check
        title_lower = title.lower()
        if "disambiguation" in title_lower or title_lower.startswith("list of "):
            return CleaningResult(text, True, f"disallowed_wikimedia_list_or_disambig_{title}", {})

        # Strip reference/external link/further reading sections
        cleaned_text = SECTION_STRIP_REGEX.sub('', text).strip()

        if len(cleaned_text.split()) < 64:
            return CleaningResult(cleaned_text, True, "post_section_strip_below_min_length", {})

        return CleaningResult(cleaned_text, False, None, {"approx_tokens": len(cleaned_text.split())})
