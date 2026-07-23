import re
from typing import List, Dict, Any, Optional
from .generic import CleaningResult

BOT_PATTERNS = [
    re.compile(r'dependabot', re.I),
    re.compile(r'stale\[bot\]', re.I),
    re.compile(r'codecov-commenter', re.I),
    re.compile(r'github-actions\[bot\]', re.I)
]

STACK_TRACE_REGEX = re.compile(r'Traceback \(most recent call last\):.*?\n\S+', re.DOTALL)
BASE64_REGEX = re.compile(r'[A-Za-z0-9+/]{100,}={0,2}')

class TechnicalCleaner:
    def clean_github_thread(
        self,
        title: str,
        body: str,
        comments: List[Dict[str, Any]],
        is_pr: bool = False
    ) -> CleaningResult:
        # Check bot title / body
        if any(b.search(title) or b.search(body) for b in BOT_PATTERNS):
            return CleaningResult("", True, "bot_generated_thread_rejected", {})

        title_tag = "<PULL_REQUEST_TITLE>" if is_pr else "<ISSUE_TITLE>"
        title_end_tag = "</PULL_REQUEST_TITLE>" if is_pr else "</ISSUE_TITLE>"
        body_tag = "<PULL_REQUEST_BODY>" if is_pr else "<ISSUE_BODY>"
        body_end_tag = "</PULL_REQUEST_BODY>" if is_pr else "</ISSUE_BODY>"

        formatted_parts = [
            f"{title_tag}\n{title.strip()}\n{title_end_tag}",
            f"{body_tag}\n{body.strip()}\n{body_end_tag}"
        ]

        turn_count = 1
        code_token_count = 0
        total_token_count = len(title.split()) + len(body.split())

        for comm in comments:
            author = comm.get("author", "")
            if any(b.search(author) for b in BOT_PATTERNS):
                continue

            comm_body = comm.get("body", "").strip()
            if not comm_body:
                continue

            # Strip base64 and huge stack traces
            comm_body = BASE64_REGEX.sub('[BASE64_BINARY_REMOVED]', comm_body)
            comm_body = STACK_TRACE_REGEX.sub('[STACK_TRACE_REMOVED]', comm_body)

            # Code token estimation
            code_blocks = re.findall(r'```.*?```', comm_body, flags=re.DOTALL)
            for cb in code_blocks:
                code_token_count += len(cb.split())

            comm_tokens = len(comm_body.split())
            total_token_count += comm_tokens

            tag = "<REVIEW_COMMENT>" if is_pr else "<COMMENT>"
            end_tag = "</REVIEW_COMMENT>" if is_pr else "</COMMENT>"
            formatted_parts.append(f"{tag}\n{comm_body}\n{end_tag}")
            turn_count += 1

        if turn_count < 2:
            return CleaningResult("", True, f"insufficient_substantive_turns_{turn_count}<2", {})

        if total_token_count > 0 and (code_token_count / total_token_count) > 0.60:
            return CleaningResult("", True, f"code_log_token_ratio_exceeded_{(code_token_count/total_token_count):.3f}>0.60", {})

        full_text = "\n\n".join(formatted_parts)
        if total_token_count < 96:
            return CleaningResult(full_text, True, f"github_thread_below_96_{total_token_count}", {})

        return CleaningResult(
            full_text,
            False,
            None,
            {"approx_tokens": total_token_count, "turn_count": turn_count, "code_ratio": code_token_count/max(total_token_count, 1)}
        )

    def clean_pep(self, text: str) -> CleaningResult:
        if not text:
            return CleaningResult("", True, "empty_text", {})

        # Strip email header boilerplate
        cleaned = re.sub(r'^(?:PEP:|Title:|Author:|Status:|Type:|Created:).*?\n', '', text, flags=re.M)
        words = cleaned.split()

        if len(words) < 256:
            return CleaningResult(cleaned, True, f"pep_below_256_{len(words)}", {})

        return CleaningResult(cleaned, False, None, {"approx_tokens": len(words)})
