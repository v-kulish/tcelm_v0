import re
from typing import Dict, List, Any, Optional
from .generic import CleaningResult

class StackExchangeCleaner:
    def __init__(self, max_site_share_quota: float = 0.10, total_se_quota_tokens: int = 300000000):
        self.max_site_share_quota = max_site_share_quota
        self.total_se_quota_tokens = total_se_quota_tokens
        self.site_token_counts: Dict[str, int] = {}

    def clean_thread(
        self,
        title: str,
        question_body: str,
        answers: List[Dict[str, Any]],
        site_name: str = "stackoverflow"
    ) -> CleaningResult:
        site_name_clean = site_name.lower().strip()
        if "meta" in site_name_clean:
            return CleaningResult("", True, f"disallowed_meta_site_{site_name}", {})

        # Sub-site quota capping
        max_site_tokens = int(self.total_se_quota_tokens * self.max_site_share_quota)
        current_site_tokens = self.site_token_counts.get(site_name_clean, 0)
        if current_site_tokens >= max_site_tokens:
            return CleaningResult("", True, f"site_quota_exceeded_{site_name_clean}", {})

        if not question_body or len(question_body.split()) < 20:
            return CleaningResult("", True, "empty_or_short_question_body", {})

        if not answers:
            return CleaningResult("", True, "no_answers_in_thread", {})

        # Format turn-structured text
        formatted_parts = [
            f"<QUESTION_TITLE>\n{title.strip()}\n</QUESTION_TITLE>",
            f"<QUESTION_BODY>\n{question_body.strip()}\n</QUESTION_BODY>"
        ]

        answer_count = 0
        comment_count = 0

        # Sort answers (e.g. accepted first, then by score)
        sorted_answers = sorted(answers, key=lambda a: (a.get('is_accepted', False), a.get('score', 0)), reverse=True)

        for ans in sorted_answers[:20]: # max 20 answers
            ans_body = ans.get('body', '').strip()
            if not ans_body or len(ans_body.split()) < 10:
                continue
            formatted_parts.append(f"<ANSWER>\n{ans_body}\n</ANSWER>")
            answer_count += 1

            for comm in ans.get('comments', [])[:40]: # max 40 comments per answer
                comm_text = comm.get('text', '').strip()
                if len(comm_text.split()) < 6 and "correct" not in comm_text.lower() and "fix" not in comm_text.lower():
                    continue
                formatted_parts.append(f"<COMMENT>\n{comm_text}\n</COMMENT>")
                comment_count += 1

        if answer_count == 0:
            return CleaningResult("", True, "no_substantive_answers_retained", {})

        full_thread_text = "\n\n".join(formatted_parts)
        tokens = full_thread_text.split()
        if len(tokens) < 128:
            return CleaningResult(full_thread_text, True, f"thread_tokens_below_128_{len(tokens)}", {})

        # Truncate if exceeds max 12,288 tokens
        if len(tokens) > 12288:
            tokens = tokens[:12288]
            full_thread_text = " ".join(tokens)

        # Update site token count
        self.site_token_counts[site_name_clean] = current_site_tokens + len(tokens)

        return CleaningResult(
            full_thread_text,
            False,
            None,
            {"approx_tokens": len(tokens), "answer_count": answer_count, "comment_count": comment_count}
        )
