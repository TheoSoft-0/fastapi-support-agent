from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.exceptions import UnexpectedModelBehavior

import main


client = TestClient(main.app)


def install_fake_search(monkeypatch, passage):
    """Return one controlled result from the local search function."""

    document = main.DOCUMENTS_BY_ID["first-steps"]

    def fake_search(query, documents, limit=3):
        assert query
        assert limit == 3

        return [
            SimpleNamespace(
                document=document,
                passages=[passage],
            )
        ]

    monkeypatch.setattr(main, "search_documents", fake_search)

    return document


def test_agent_search_with_no_results_abstains(monkeypatch):
    search_calls = []

    def fake_search(query, documents, limit=3):
        search_calls.append(query)
        return []

    async def fake_retrieval_run(question, *, deps):
        tool_result = main.search_documentation(question, deps)
        assert "No relevant" in tool_result

        return SimpleNamespace(
            output="No relevant documentation was found."
        )

    async def answer_agent_must_not_run(*args, **kwargs):
        pytest.fail("The answer agent must not run without evidence.")

    monkeypatch.setattr(main, "search_documents", fake_search)
    monkeypatch.setattr(main.support_agent, "run", fake_retrieval_run)
    monkeypatch.setattr(
        main.answer_agent,
        "run",
        answer_agent_must_not_run,
    )

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
        "The agent searched local documentation but found no relevant passages."
    )


def test_verified_retrieved_quote_is_returned(monkeypatch):
    quote = "The simplest FastAPI file could look like this:"
    document = install_fake_search(monkeypatch, quote)

    async def fake_retrieval_run(question, *, deps):
        tool_result = main.search_documentation(question, deps)
        assert quote in tool_result

        return SimpleNamespace(
            output="Retrieved the first-steps documentation."
        )

    async def fake_answer_run(prompt):
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

    monkeypatch.setattr(main.support_agent, "run", fake_retrieval_run)
    monkeypatch.setattr(main.answer_agent, "run", fake_answer_run)

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

    async def fake_retrieval_run(question, *, deps):
        main.search_documentation(question, deps)

        return SimpleNamespace(
            output="Retrieved the first-steps documentation."
        )

    async def fake_answer_run(prompt):
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

    monkeypatch.setattr(main.support_agent, "run", fake_retrieval_run)
    monkeypatch.setattr(main.answer_agent, "run", fake_answer_run)

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


def test_missing_agent_tool_call_uses_application_fallback(monkeypatch):
    quote = "The simplest FastAPI file could look like this:"
    document = main.DOCUMENTS_BY_ID["first-steps"]

    search_queries = []
    observed_dependencies = []

    def fake_search(query, documents, limit=3):
        search_queries.append(query)

        return [
            SimpleNamespace(
                document=document,
                passages=[quote],
            )
        ]

    async def fake_retrieval_run(question, *, deps):
        observed_dependencies.append(deps)

        # Simulate the agent finishing without calling its search tool.
        return SimpleNamespace(
            output="No documentation search was performed."
        )

    async def fake_answer_run(prompt):
        assert quote in prompt

        return SimpleNamespace(
            output=main.ModelDraft(
                can_answer=True,
                answer=(
                    "A basic GET endpoint uses a route decorator.\n\n"
                    "```python\n"
                    "from fastapi import FastAPI\n\n"
                    "app = FastAPI()\n\n"
                    '@app.get("/")\n'
                    "async def root():\n"
                    '    return {"message": "Hello World"}\n'
                    "```"
                ),
                evidence_quote=quote,
                source_document_id=document.id,
                reason_if_not_answerable=None,
            )
        )

    monkeypatch.setattr(main, "search_documents", fake_search)
    monkeypatch.setattr(main.support_agent, "run", fake_retrieval_run)
    monkeypatch.setattr(main.answer_agent, "run", fake_answer_run)

    question = "How do I create a basic GET endpoint in FastAPI?"

    response = client.post(
        "/support",
        json={"question": question},
    )

    assert response.status_code == 200
    assert search_queries == [question]

    dependencies = observed_dependencies[0]

    assert dependencies.agent_tool_calls == 0
    assert dependencies.application_fallback_used is True
    assert dependencies.search_calls == 1

    body = response.json()

    assert body["answer"] is not None
    assert body["evidence_sufficient"] is True
    assert body["source"]["title"] == document.title
    assert body["source"]["url"] == document.url
    assert body["source"]["quote"] == quote


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "How do I configure CORS middleware in FastAPI?",
            True,
        ),
        (
            "How do I create a WebSocket endpoint in FastAPI?",
            True,
        ),
        (
            "How do I test FastAPI endpoints using TestClient?",
            True,
        ),
        (
            "Is a JWT's information encrypted?",
            True,
        ),
        (
            "How do I implement quantum teleportation for zebras?",
            False,
        ),
    ],
)
def test_documentation_search_scope(question, expected):
    assert main.question_requires_documentation_search(question) is expected


def test_invalid_model_output_returns_502(monkeypatch):
    async def fake_retrieval_run(question, *, deps):
        main.search_documentation(question, deps)

        return SimpleNamespace(
            output="Retrieved the first-steps documentation."
        )

    async def fake_answer_run(prompt):
        raise UnexpectedModelBehavior("Invalid structured output")

    monkeypatch.setattr(main.support_agent, "run", fake_retrieval_run)
    monkeypatch.setattr(main.answer_agent, "run", fake_answer_run)

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


def test_health_endpoint_returns_200():
    response = client.get("/health")

    assert response.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": 123},
        {"question": "abcd"},
        {"question": "     "},
        {"question": "a" * 1001},
    ],
)
def test_invalid_support_questions_return_422(payload):
    response = client.post(
        "/support",
        json=payload,
    )

    assert response.status_code == 422
