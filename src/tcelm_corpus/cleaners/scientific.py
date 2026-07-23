import re
from typing import Dict, Optional, List
from .generic import CleaningResult

SECTION_REMOVE_REGEX = re.compile(
    r'\n(?:#+\s*|\\section\*?\{?)?\s*(?:References|Bibliography|Acknowledgements|Funding|Author\s+Contributions|Supplementary\s+Material).*$',
    re.DOTALL | re.I
)

LATEX_MATH_DISPLAY_REGEX = re.compile(r'\\\[(.*?)\\\]|\$\$(.*?)\$\$|\\begin\{equation\}(.*?)\\end\{equation\}', re.DOTALL)
LATEX_MATH_INLINE_REGEX = re.compile(r'\\\((.*?)\\\)|\$([^\$\n]+)\$')

class ScientificCleaner:
    def __init__(self):
        self.field_sample_counts: Dict[str, int] = {}

    def format_equations(self, text: str) -> str:
        # Replace display math with <EQUATION> tags
        def replace_display(match):
            eq_text = match.group(1) or match.group(2) or match.group(3) or ""
            return f"\n<EQUATION>\n{eq_text.strip()}\n</EQUATION>\n"

        text = LATEX_MATH_DISPLAY_REGEX.sub(replace_display, text)
        return text

    def clean_arxiv(self, text: str) -> CleaningResult:
        if not text:
            return CleaningResult("", True, "empty_text", {})

        # Strip bibliography / acknowledgements / author affiliations
        cleaned = SECTION_REMOVE_REGEX.sub('', text)

        # Format equations cleanly
        cleaned = self.format_equations(cleaned)

        words = cleaned.split()
        if len(words) < 512:
            return CleaningResult(cleaned, True, f"arxiv_below_512_{len(words)}", {})

        return CleaningResult(cleaned, False, None, {"approx_tokens": len(words)})

    def clean_pes2o(self, text: str, field_family: str = "cs_engineering") -> CleaningResult:
        if not text:
            return CleaningResult("", True, "empty_text", {})

        cleaned = SECTION_REMOVE_REGEX.sub('', text)
        words = cleaned.split()

        if len(words) < 512:
            return CleaningResult(cleaned, True, f"pes2o_below_512_{len(words)}", {})

        self.field_sample_counts[field_family] = self.field_sample_counts.get(field_family, 0) + len(words)
        return CleaningResult(cleaned, False, None, {"approx_tokens": len(words), "field_family": field_family})

    def clean_pubmed(self, text: str) -> CleaningResult:
        if not text:
            return CleaningResult("", True, "empty_text", {})

        cleaned = SECTION_REMOVE_REGEX.sub('', text)
        words = cleaned.split()

        if len(words) < 256:
            return CleaningResult(cleaned, True, f"pubmed_below_256_{len(words)}", {})

        return CleaningResult(cleaned, False, None, {"approx_tokens": len(words)})
