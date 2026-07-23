from typing import Dict, Any
from .base import BaseSourceAdapter
from .stackexchange import StackExchangeAdapter
from .technical import TechnicalAdapter
from .scientific import ScientificAdapter
from .books import BooksAdapter, WikimediaAdapter, CCCCAdapter, EducationalAdapter, GovernmentLegalAdapter

def get_source_adapter(source_name: str, config: Dict[str, Any]) -> BaseSourceAdapter:
    if "stackexchange" in source_name:
        return StackExchangeAdapter(source_name, config)
    elif "github" in source_name or "pep" in source_name or "python_enhancement" in source_name:
        return TechnicalAdapter(source_name, config)
    elif "arxiv" in source_name or "pes2o" in source_name or "pubmed" in source_name:
        return ScientificAdapter(source_name, config)
    elif "gutenberg" in source_name or "doab" in source_name or "pre_1929" in source_name or "pressbooks" in source_name:
        return BooksAdapter(source_name, config)
    elif "wikimedia" in source_name:
        return WikimediaAdapter(source_name, config)
    elif "cccc" in source_name:
        return CCCCAdapter(source_name, config)
    elif "libretexts" in source_name or "oercommons" in source_name:
        return EducationalAdapter(source_name, config)
    elif "hansard" in source_name or "usgpo" in source_name or "regulations" in source_name or "caselaw" in source_name:
        return GovernmentLegalAdapter(source_name, config)
    else:
        return CCCCAdapter(source_name, config)
