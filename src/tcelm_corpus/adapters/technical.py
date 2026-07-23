from typing import Dict, Any, Tuple
from .base import BaseSourceAdapter

class TechnicalAdapter(BaseSourceAdapter):
    def extract_record(self, raw_record: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        license_status = raw_record.get("license", raw_record.get("licence", "")) or "missing"

        if "python_enhancement_proposals" in self.source_name:
            text = raw_record.get("text", "") or raw_record.get("content", "")
            title = raw_record.get("title", "PEP")
            metadata = {"title": title, "license": license_status, "type": "pep"}
            return text, license_status, metadata

        # GitHub Archive issues / PRs
        title = (raw_record.get("title") or "").strip()
        body = (raw_record.get("body") or raw_record.get("text") or "").strip()
        comments = raw_record.get("comments", [])
        is_pr = raw_record.get("is_pull_request", False)

        title_tag = "<PULL_REQUEST_TITLE>" if is_pr else "<ISSUE_TITLE>"
        title_end_tag = "</PULL_REQUEST_TITLE>" if is_pr else "</ISSUE_TITLE>"
        body_tag = "<PULL_REQUEST_BODY>" if is_pr else "<ISSUE_BODY>"
        body_end_tag = "</PULL_REQUEST_BODY>" if is_pr else "</ISSUE_BODY>"

        parts = [
            f"{title_tag}\n{title}\n{title_end_tag}",
            f"{body_tag}\n{body}\n{body_end_tag}"
        ]

        if isinstance(comments, list):
            for comm in comments:
                if isinstance(comm, dict):
                    comm_text = comm.get("body", "").strip()
                    if comm_text:
                        tag = "<REVIEW_COMMENT>" if is_pr else "<COMMENT>"
                        end_tag = "</REVIEW_COMMENT>" if is_pr else "</COMMENT>"
                        parts.append(f"{tag}\n{comm_text}\n{end_tag}")

        text = "\n\n".join(parts) if body else raw_record.get("text", "")

        metadata = {
            "title": title,
            "url": raw_record.get("url", ""),
            "repo": raw_record.get("repo", ""),
            "license": license_status
        }

        return text, license_status, metadata
