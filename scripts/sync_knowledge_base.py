import httpx
import json
import re

from bs4 import BeautifulSoup
from datetime import date
from pathlib import Path





def extract_content(root) -> str:
    """Extract readable documentation while preserving code formatting."""
    block_tags = (
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "li", "pre", "blockquote", "dt", "dd", "table",
    )

    # Remove non-content elements from the selected documentation area.
    for element in root.select("script, style, noscript, svg"):
        element.decompose()

    blocks = []

    for element in root.find_all(block_tags):
        # Avoid extracting the same text twice from nested elements.
        if element.find_parent(list(block_tags)):
            continue

        if element.name == "pre":
            # Preserve original whitespace inside code examples.
            text = element.get_text(separator="", strip=False).strip()

        elif element.name == "table":
            rows = []

            for row in element.find_all("tr"):
                cells = [
                    cell.get_text(" ", strip=True)
                    for cell in row.find_all(
                        ["th", "td"],
                        recursive=False,
                    )
                ]

                if cells:
                    rows.append(" | ".join(cells))

            text = "\n".join(rows)

        else:
            # Join inline HTML fragments with spaces instead of newlines.
            text = element.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()

        if text:
            blocks.append(text)

    return "\n\n".join(blocks)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "knowledge_base" / "documents.json"

# This is an explicit allowlist. The script does not discover or crawl
# additional pages.
DOCUMENT_SOURCES = [
    {
        "id": "first-steps",
        "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
        "keywords": [
            "application", "endpoint", "route", "GET", "POST",
            "path operation", "create app", "run server",
        ],
    },
    {
        "id": "path-parameters",
        "url": "https://fastapi.tiangolo.com/tutorial/path-params/",
        "keywords": [
            "path parameter", "URL path", "path variable",
            "parameter validation", "integer", "type conversion",
        ],
    },
    {
        "id": "query-parameters",
        "url": "https://fastapi.tiangolo.com/tutorial/query-params/",
        "keywords": [
            "query parameter", "optional parameter", "default value",
            "URL query", "skip", "limit",
        ],
    },
    {
        "id": "request-body",
        "url": "https://fastapi.tiangolo.com/tutorial/body/",
        "keywords": [
            "request body", "Pydantic", "BaseModel", "JSON",
            "POST request", "body validation",
        ],
    },
    {
        "id": "response-model",
        "url": "https://fastapi.tiangolo.com/tutorial/response-model/",
        "keywords": [
            "response model", "response schema", "return type",
            "output validation", "response filtering",
        ],
    },
    {
        "id": "handling-errors",
        "url": "https://fastapi.tiangolo.com/tutorial/handling-errors/",
        "keywords": [
            "HTTPException", "exceptions", "raise",
            "HTTP error", "error response",
        ],
    },
    {
        "id": "response-status-code",
        "url": "https://fastapi.tiangolo.com/tutorial/response-status-code/",
        "keywords": [
            "status code", "HTTP response", "201", "404",
            "response status",
        ],
    },
    {
        "id": "dependencies",
        "url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
        "keywords": [
            "Depends", "dependency injection", "shared logic",
            "database session", "dependencies",
        ],
    },
    {
        "id": "oauth2-jwt",
        "url": "https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/",
        "keywords": [
            "OAuth2", "password", "Bearer", "JWT",
            "authentication", "authorization", "security",
        ],
    },
]


def extract_document(
    client: httpx.Client,
    source: dict[str, object],
) -> dict[str, object]:
    """Download one approved page and extract its main article."""
    url = str(source["url"])
    response = client.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    article = (
        soup.select_one("article.md-content__inner")
        or soup.find("article")
        or soup.find("main")
    )

    if article is None:
        raise RuntimeError(f"Could not locate article content at {url}")

    # Remove page elements that do not belong in the knowledge base.
    for element in article.select(  "script, style, nav, button, .headerlink, .md-clipboard" ):
        element.decompose()

    heading = article.find("h1")
    if heading is None:
        raise RuntimeError(f"Could not find a title at {url}")

    title = re.sub(r"\s+", " ", heading.get_text(" ", strip=True))
    title = title.removesuffix("¶").strip()

    content = extract_content(article)

    if len(content) < 200:
        raise RuntimeError(
            f"Extracted too little content from {url}: "
            f"{len(content)} characters"
        )

    return {
        "id": source["id"],
        "title": title,
        "url": url,
        "keywords": source["keywords"],
        "content": content,
    }


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    documents = []

    with httpx.Client(
        timeout=30.0,
        follow_redirects=True,
        headers={
            "User-Agent": "FastAPI-Support-Agent/0.1 documentation snapshot"
        },
    ) as client:
        for source in DOCUMENT_SOURCES:
            document = extract_document(client, source)
            documents.append(document)

            word_count = len(str(document["content"]).split())
            print(
                f"Downloaded {document['id']}: "
                f"{document['title']} ({word_count} words)"
            )

    snapshot = {
        "schema_version": 1,
        "snapshot_date": date.today().isoformat(),
        "source_name": "Official FastAPI documentation",
        "source_license": "MIT",
        "source_license_url": (
            "https://github.com/fastapi/fastapi/blob/master/LICENSE"
        ),
        "documents": documents,
    }

    OUTPUT_FILE.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved {len(documents)} documents to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
