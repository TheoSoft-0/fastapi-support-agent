
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.exceptions import UnexpectedModelBehavior

import main


client = TestClient(main.app)


def install_fake_search(monkeypatch, passage):
    """Return one controlled result from the local search function."""
    document = main.DOCUMENTS_BY_ID["first-steps"]

    def fake_search(question, documents, limit=3):
        assert question
        assert limit == 3

        return [
            SimpleNamespace(
                document=document,
                passages=[passage],
            )
        ]

    monkeypatch.setattr(main, "search_documents", fake_search)

    return document


def test_no_retrieval_results_abstains_without_calling_model(monkeypatch):
    search_calls = []

    def fake_search(question, documents, limit=3):
        search_calls.append(question)
        return []

    async def model_must_not_run(*args, **kwargs):
        pytest.fail("The model should not run without retrieved evidence.")

    monkeypatch.setattr(main, "search_documents", fake_search)
    monkeypatch.setattr(main.support_agent, "run", model_must_not_run)

    question = "How do I implement quantum teleportation for zebras?"

    response = client.post(
        "/support",
        json={"question": question},
    )

    assert response.status_code == 200
    assert search_calls == [question]

    body = response.json()

    assert body["answer"] is None
    assert body["evidence_sufficient"] is False
    assert body["source"] is None
    assert body["explanation"] == (
        "No relevant local documentation passages were found."
    )


def test_verified_retrieved_quote_is_returned(monkeypatch):
    quote = "The simplest FastAPI file could look like this:"
    document = install_fake_search(monkeypatch, quote)

    async def fake_run(prompt):
        assert quote in prompt

        return SimpleNamespace(
            output=main.ModelDraft(
                can_answer=True,
                answer=(
                    "Create a FastAPI application with a GET endpoint. "
                    "For example:\n\n"
                    "```python\n"
                    "from fastapi import FastAPI\n\n"
                    "app = FastAPI()\n\n"
                    "@app.get(\"/\")\n"
                    "async def root():\n"
                    "    return {\"message\": \"Hello World\"}\n"
                    "```"
                ),
                evidence_quote=quote,
                source_document_id=document.id,
                reason_if_not_answerable=None,
            )
        )

    monkeypatch.setattr(main.support_agent, "run", fake_run)

    response = client.post(
        "/support",
        json={
            "question": "How do I create a basic GET endpoint in FastAPI?"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] is not None
    assert body["evidence_sufficient"] is True
    assert body["source"]["title"] == document.title
    assert body["source"]["url"] == document.url
    assert body["source"]["quote"] == quote


def test_unverifiable_quote_is_rejected(monkeypatch):
    quote = "The simplest FastAPI file could look like this:"
    document = install_fake_search(monkeypatch, quote)

    async def fake_run(prompt):
        return SimpleNamespace(
            output=main.ModelDraft(
                can_answer=True,
                answer=(
                    "Create a FastAPI application with a GET endpoint. "
                    "For example:\n\n"
                    "```python\n"
                    "from fastapi import FastAPI\n\n"
                    "app = FastAPI()\n\n"
                    '@app.get("/")\n'
                    "async def root():\n"
                    '    return {"message": "Hello World"}\n'
                    "```"
                ),
                evidence_quote="You can teleport zebras with FastAPI.",
                source_document_id=document.id,
                reason_if_not_answerable=None,
            )
        )

    monkeypatch.setattr(main.support_agent, "run", fake_run)

    response = client.post(
        "/support",
        json={
            "question": "How do I create a basic GET endpoint in FastAPI?"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] is None
    assert body["evidence_sufficient"] is False
    assert body["source"] is None
    assert "quotation could not be verified" in body["explanation"]


def test_invalid_model_output_returns_502(monkeypatch):
    quote = "The simplest FastAPI file could look like this:"
    install_fake_search(monkeypatch, quote)

    async def fake_run(prompt):
        raise UnexpectedModelBehavior("Invalid structured output")

    monkeypatch.setattr(main.support_agent, "run", fake_run)

    response = client.post(
        "/support",
        json={
            "question": "How do I create a basic GET endpoint in FastAPI?"
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The local model returned invalid structured output."
    )
