# FastAPI Support Agent

A locally running AI support agent for answering questions about FastAPI using retrieved official documentation.

The project demonstrates agent-directed tool use, document retrieval, structured AI outputs, evidence verification, Python code validation, deterministic recovery, automated testing, and continuous integration.

The application runs locally using **Ollama**, **Qwen3 4B Instruct**, **PydanticAI**, and **FastAPI**. It does not require a paid AI API.

## Features

- **Agent-directed retrieval:** A PydanticAI agent decides when documentation searches are needed and selects search queries.
- **Local knowledge base:** Retrieves information from a local snapshot of nine official FastAPI documentation pages.
- **Evidence-based answers:** A separate answer agent receives retrieved passages and produces a structured response.
- **Citation verification:** Checks that the selected source document was retrieved and that its quotation exists in the original documentation.
- **Python code validation:** Uses Python's Abstract Syntax Tree (AST) to inspect generated examples. AST analysis checks code structure without executing the code.
- **Deterministic recovery:** Uses predefined, validated code templates when generated examples fail validation.
- **Controlled correction:** Allows one correction-agent attempt when no suitable deterministic template exists.
- **Safe abstention:** Declines to answer when sufficient documentation evidence cannot be retrieved or verified.
- **Search failure handling:** Records search errors and prevents unnecessary fallback searches after failures.
- **Automated testing:** Includes 60 automated tests.
- **Continuous integration:** GitHub Actions installs locked dependencies, compiles Python files, and runs the test suite.

## Architecture

The application separates documentation retrieval from answer generation.

```text
                 User
                   |
                   v
             FastAPI API
             POST /support
                   |
                   v
          Support Dependencies
          Request-specific state
                   |
                   v
          Documentation Agent
          (PydanticAI)
                   |
             Search tool
                   |
                   v
          Local Retrieval System
                   |
                   v
          Retrieved Documentation
                   |
                   v
             Answer Agent
          (Structured ModelDraft)
                   |
                   v
           Python Validation
           Citation Validation
                   |
          +--------+--------+
          |                 |
       Valid             Invalid
          |                 |
          |          Deterministic Template
          |                 |
          |          Correction Agent
          |          when applicable
          |                 |
          +--------+--------+
                   |
                   v
          Final Validation
                   |
                   v
          SupportResponse
```

### 1. Documentation retrieval agent

The first agent receives the user's question.

It decides whether to search the local documentation collection. It can call the registered `search_fastapi_documentation` tool.

The tool searches the local knowledge base. It returns relevant passages and records them for subsequent verification.

Python also provides a deterministic fallback search when the agent skips retrieval for a recognized FastAPI-related question.

### 2. Local retrieval system

The retrieval system is implemented in `retrieval.py`.

It uses weighted keyword matching:

| Matching location | Score |
| ----------------- | ----: |
| Document keywords |     6 |
| Document title    |     4 |
| Document content  |     1 |

A document must reach a minimum relevance score of 5.

The system returns up to three documents. It extracts relevant text passages with surrounding context.

This is keyword-based retrieval. It does not use vector embeddings or a separate vector database.

### 3. Structured answer generation

The second agent receives the original question and retrieved documentation.

It produces a structured Pydantic model called `ModelDraft`.

The model specifies whether an answer is supported. It also supplies the answer, quotation, and source document ID.

The answer agent is instructed to use only the supplied documentation.

### 4. Evidence and citation verification

The application performs deterministic checks after answer generation.

It verifies that:

- The selected document exists in the local knowledge base.
- The selected document was actually retrieved for the current question.
- The document matches the expected topic when a specific source is required.
- The supporting quotation matches retrieved evidence and the original document.

An answer with an invalid source or unverifiable quotation is rejected.

### 5. Python code validation and recovery

Python examples are extracted from generated answers and validated.

The validator uses Python's AST. This checks Python syntax and expected FastAPI code structures without executing generated code.

When validation fails, the application first attempts to use a predefined deterministic template.

If no suitable template exists, the correction agent receives one correction attempt. The corrected output is validated again.

If validation still fails, the application abstains.

### 6. Search failure handling

Documentation search failures are handled separately from searches that return no results.

The application records safe error descriptions. It logs the underlying exception and avoids exposing internal exception details.

A failed search causes an abstention. It does not trigger another automatic search.

## Knowledge Base

The knowledge base is stored in:

`knowledge_base/documents.json`

It contains nine pages from the official FastAPI documentation:

1. First steps and application creation.
2. Path parameters.
3. Query parameters.
4. Request bodies and Pydantic models.
5. Response models.
6. Handling errors and `HTTPException`.
7. HTTP response status codes.
8. Dependency injection.
9. OAuth2 and JSON Web Tokens (JWT).

The knowledge-base synchronization script is:

`scripts/sync_knowledge_base.py`

It downloads pages from an explicit allowlist. It does not crawl the entire FastAPI website.

The script extracts readable article content. It preserves code examples and relevant document metadata.

The resulting snapshot is saved locally. Normal application startup reads this snapshot without downloading documentation.

To refresh the snapshot:

```bash
uv run python scripts/sync_knowledge_base.py
```

This operation requires an internet connection.

## Technology Stack

| Technology        | Purpose                                   |
| ----------------- | ----------------------------------------- |
| Python 3.13       | Application language                      |
| FastAPI           | HTTP API framework                        |
| Pydantic          | Data validation and structured models     |
| PydanticAI        | Agent orchestration and tool registration |
| Ollama            | Local model runtime                       |
| Qwen3 4B Instruct | Local language model                      |
| BeautifulSoup     | HTML documentation extraction             |
| httpx             | HTTP requests                             |
| pytest            | Automated testing                         |
| uv                | Dependency and environment management     |
| GitHub Actions    | Continuous integration                    |

The model uses the local Ollama OpenAI-compatible endpoint:

`http://localhost:11434/v1`

The configured model is:

`qwen3:4b-instruct`

## Installation

### Prerequisites

Install the following software:

- Python 3.13.
- [uv](https://docs.astral.sh/uv/).
- [Ollama](https://ollama.com/).

The project uses a local language model. Ollama must be running before using the support endpoint.

### 1. Clone the repository

```bash
git clone https://github.com/TheoSoft-0/fastapi-support-agent.git

cd fastapi-support-agent
```

### 2. Install dependencies

```bash
uv sync --locked --all-groups --no-install-project
```

This creates a local virtual environment and installs the dependencies recorded in `uv.lock`.

### 3. Download the configured model

```bash
ollama pull qwen3:4b-instruct
```

The model runs locally through Ollama.

### 4. Start Ollama

Ensure that Ollama is running and listening on port `11434`.

On systems where Ollama is not already running as a background service, start it with:

```bash
ollama serve
```

You can verify the downloaded model with:

```bash
ollama list
```

### 5. Start the application

From the project directory:

```bash
uv run uvicorn main:app --reload
```

The API will be available at:

`http://127.0.0.1:8000`

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

## API Usage

### Health check

**Endpoint:** `GET /health`

Example:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

### Submit a support question

**Endpoint:** `POST /support`

Request:

```bash
curl -X POST http://127.0.0.1:8000/support \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I create a basic GET endpoint in FastAPI?"
  }'
```

The `question` field must contain between 5 and 1000 characters. Whitespace-only questions are rejected.

A successful, evidence-supported response has this structure:

```json
{
  "question": "How do I create a basic GET endpoint in FastAPI?",
  "answer": "The validated answer appears here.",
  "evidence_sufficient": true,
  "explanation": null,
  "source": {
    "title": "First Steps",
    "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "quote": "The verified source quotation appears here."
  }
}
```

When sufficient evidence is unavailable, the application abstains:

```json
{
  "question": "The original question",
  "answer": null,
  "evidence_sufficient": false,
  "explanation": "The reason an answer could not be verified.",
  "source": null
}
```

### HTTP error responses

| Status | Meaning                                                 |
| ------ | ------------------------------------------------------- |
| 200    | Request processed, including evidence-based abstentions |
| 422    | Invalid request data                                    |
| 502    | Local model returned invalid structured output          |
| 504    | Local model processing timed out                        |

## Testing

The automated tests do not require Ollama. They use controlled test doubles for agent and model interactions.

Run the complete test suite:

```bash
uv run python -m pytest -q
```

Compile the application and evaluation script:

```bash
uv run python -m py_compile \
  main.py \
  tests/test_api.py \
  scripts/evaluate_support_agent.py
```

The test suite covers:

- API request validation and response behaviour.
- Documentation retrieval and relevance ranking.
- Agent-directed search and application fallback.
- Evidence and citation verification.
- Python syntax and structural validation.
- Deterministic code templates.
- Search failures and safe abstention.
- Evaluation instrumentation.

## Model Evaluation

The project includes a model evaluation script:

`scripts/evaluate_support_agent.py`

It runs 18 predefined evaluation cases against the locally running application.

The evaluation covers supported questions, rephrased questions, unsupported topics, retrieval decisions, documentation evidence, citations, answer generation, and code validation.

Run it with:

```bash
uv run python scripts/evaluate_support_agent.py
```

Ollama must be running, and `qwen3:4b-instruct` must be available locally.

### Recorded evaluation results

The recorded evaluation on November 3, 2026, produced:

| Measurement                           | Result |
| ------------------------------------- | -----: |
| Evaluation cases passed               |  18/18 |
| Cases using agent-directed tools      |  14/18 |
| Cases using application fallback      |   3/18 |
| Cases correctly skipping search       |   1/18 |
| Documentation search operations       |     17 |
| Initial model drafts returned         |  10/10 |
| Initial code validation passed        |   0/10 |
| Deterministic fallback templates used |  10/10 |

All 18 evaluation cases passed.

The code-generation results require an important qualification.

The local Qwen model produced invalid Python examples in all ten coding cases during that evaluation run.

The deterministic validation and template recovery mechanisms corrected those examples.

Therefore, the evaluation demonstrates reliable validation and recovery for the tested cases. It does not demonstrate reliable standalone Python code generation by the language model.

The results describe one evaluation run. They should not be interpreted as a guarantee that every future model response will behave identically.

## Continuous Integration

The repository includes a GitHub Actions workflow:

`.github/workflows/ci.yml`

It runs automatically on pushes and pull requests targeting `main`.

The workflow:

1. Checks out the repository.
2. Installs Python 3.13.
3. Installs uv.
4. Installs dependencies using the committed lockfile.
5. Compiles the application and evaluation script.
6. Runs the complete pytest suite.

The CI workflow does not start Ollama or run the model evaluation. This keeps automated testing independent of local model availability.

The successful GitHub Actions run for commit `b6f576d` completed with:

```text
60 passed in 3.71s
```

## Project Structure

```text
fastapi-support-agent/
│
├── main.py
│
├── retrieval.py
│
├── code_validation.py
│
├── pyproject.toml
├── uv.lock
│
├── knowledge_base/
│   └── documents.json
│
├── scripts/
│   ├── sync_knowledge_base.py
│   └── evaluate_support_agent.py
│
├── tests/
│   ├── evaluation_cases.json
│   ├── test_api.py
│   ├── test_code_templates_runtime.py
│   ├── test_code_validation.py
│   ├── test_retrieval.py
│   └── test_semantic_evaluation.py
│
└── .github/
    └── workflows/
        └── ci.yml
```

### Main application files

| File                                | Responsibility                                                                                 |
| ----------------------------------- | ---------------------------------------------------------------------------------------------- |
| `main.py`                           | FastAPI endpoints, PydanticAI agents, tool execution, evidence checks, recovery, and responses |
| `retrieval.py`                      | Knowledge-base loading, keyword relevance scoring, and passage extraction                      |
| `code_validation.py`                | Python syntax checks, FastAPI structure validation, and deterministic code templates           |
| `knowledge_base/documents.json`     | Local snapshot of official FastAPI documentation                                               |
| `scripts/sync_knowledge_base.py`    | Downloads and extracts approved documentation pages                                            |
| `scripts/evaluate_support_agent.py` | Runs and reports the 18-case evaluation                                                        |
| `tests/`                            | Automated unit, API, and evaluation tests                                                      |
| `.github/workflows/ci.yml`          | Automated GitHub Actions checks                                                                |

## Current Limitations

- The knowledge base covers nine selected documentation pages, not all FastAPI functionality.
- Retrieval uses keyword matching rather than semantic vector search.
- The local 4B model frequently generates invalid Python code.
- Deterministic recovery covers only predefined supported code structures.
- The evaluation contains 18 predefined cases. Broader evaluation is still needed.
- The evaluation results represent a local model run, not a guarantee of future performance.
- The correction agent is a limited recovery mechanism. It does not guarantee that arbitrary generated code can be repaired.
- The application currently supports local Ollama inference rather than a configurable range of model providers.

## Project Goals

This project was built to develop and demonstrate practical AI engineering skills.

The main areas are agent-directed tool use, local inference, retrieval systems, structured model outputs, deterministic validation, error handling, evaluation, testing, and reproducible deployment checks.

The implementation deliberately separates probabilistic model behaviour from deterministic application validation. This makes it possible to reject unsupported answers and recover from invalid code instead of trusting model output blindly.
