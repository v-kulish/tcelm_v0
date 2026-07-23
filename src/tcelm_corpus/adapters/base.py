from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List

class BaseSourceAdapter(ABC):
    """
    Abstract contract for source-specific structure extraction, license handling,
    and provenance normalization before generic cleaning.
    """
    def __init__(self, source_name: str, config: Dict[str, Any]):
        self.source_name = source_name
        self.config = config

    @abstractmethod
    def extract_record(self, raw_record: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Extracts (structured_text, license_status, metadata_dict) from raw HF record.
        If license is missing, returns license_status="missing".
        """
        pass
