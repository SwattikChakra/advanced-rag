"""
document.py — shared Document model.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    text: str
    source: str
    page: int = -1
    chunk_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Document(source={self.source!r}, page={self.page}, id={self.chunk_id}, text={preview!r}...)"
