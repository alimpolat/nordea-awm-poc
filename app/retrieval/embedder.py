"""gemini-embedding-001 wrapper with retrieval task-types.

Documents are embedded with RETRIEVAL_DOCUMENT, queries with RETRIEVAL_QUERY —
asymmetric embedding measurably improves retrieval. Vectors are MRL-truncated to
768 dimensions (the dim used by the Qdrant collection).
"""
from typing import Literal

from google.genai import types

from app.llm.vertex_client import client
from app.settings import settings

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


def embed(texts: list[str], task_type: TaskType, *, dim: int = 768) -> list[list[float]]:
    """Return one `dim`-length embedding per input text."""
    cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim)
    resp = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=texts,
        config=cfg,
    )
    return [list(e.values) for e in resp.embeddings]
