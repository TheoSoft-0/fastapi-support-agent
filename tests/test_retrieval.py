from pathlib import Path

import pytest

from retrieval import (
    DEFAULT_KNOWLEDGE_BASE,
    Document,
    load_documents,
    search_documents,
)


def test_loads_local_documentation_snapshot():
    documents = load_documents()

    assert len(documents) == 9
    assert len({document.id for document in documents}) == 9
    assert all(document.content.strip() for document in documents)


def test_oauth_question_finds_oauth_document():
    documents = load_documents()

    results = search_documents(
        "How do I configure OAuth2 password authentication?",
        documents,
    )

    assert results
    assert results[0].document.id == "oauth2-jwt"
    assert results[0].passages


def test_get_endpoint_question_finds_first_steps():
    documents = load_documents()

    results = search_documents(
        "How do I create a basic GET endpoint in FastAPI?",
        documents,
    )

    assert results
    assert results[0].document.id == "first-steps"
    assert results[0].passages


def test_unrelated_question_returns_no_results():
    documents = load_documents()

    results = search_documents(
        "zebra quantum teleportation",
        documents,
    )

    assert results == []


def test_empty_query_returns_no_results():
    documents = load_documents()

    assert search_documents("the and a", documents) == []


def test_invalid_limit_is_rejected():
    documents = load_documents()

    with pytest.raises(ValueError, match="limit"):
        search_documents("OAuth2 authentication", documents, limit=0)
        
        

def test_weak_partial_match_is_filtered_out():
    documents = load_documents()

    results = search_documents(
        "How do I implement quantum teleportation for zebras?",
        documents,
        limit=3,
    )

    assert results == []


def test_retrieved_passage_contains_surrounding_context():
    documents = load_documents()

    results = search_documents(
        "How do I create a basic GET endpoint in FastAPI?",
        documents,
        limit=3,
    )

    first_steps = next(
        (
            result
            for result in results
            if result.document.id == "first-steps"
        ),
        None,
    )

    assert first_steps is not None
    assert any(
        len(passage) > 80
        for passage in first_steps.passages
    )


def test_websocket_question_returns_no_results():
    documents = load_documents()

    results = search_documents(
        "How do I create a WebSocket endpoint in FastAPI?",
        documents,
    )

    assert results == []

