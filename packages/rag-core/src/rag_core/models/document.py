from typing import Any

from pydantic import BaseModel, Field


class ParsedDocument(BaseModel):
    """Text and metadata extracted from a source document."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
