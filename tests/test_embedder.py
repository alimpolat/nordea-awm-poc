"""Live tests for the gemini-embedding-001 task-typed wrapper.

These call Vertex AI. Marked `live` so they can be deselected with `-m "not live"`.
"""
import pytest

from app.retrieval.embedder import embed

pytestmark = pytest.mark.live


def test_embed_returns_768_dim_vector():
    vecs = embed(["Bergström holds Nordic equity and Gulf real estate."], "RETRIEVAL_DOCUMENT")
    assert len(vecs) == 1
    assert len(vecs[0]) == 768
    assert all(isinstance(x, float) for x in vecs[0][:5])


def test_task_type_changes_the_embedding():
    text = "Nordic equity concentration risk"
    doc = embed([text], "RETRIEVAL_DOCUMENT")[0]
    qry = embed([text], "RETRIEVAL_QUERY")[0]
    assert doc != qry  # same text, different task type -> different vector


def test_batch_returns_one_vector_per_input():
    vecs = embed(["alpha", "beta", "gamma"], "RETRIEVAL_DOCUMENT")
    assert len(vecs) == 3
    assert all(len(v) == 768 for v in vecs)
