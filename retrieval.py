import json
import re
from dataclasses import dataclass
from typing import Sequence
from pathlib import Path

MIN_RELEVANCE_SCORE = 5

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_KNOWLEDGE_BASE = ( PROJECT_ROOT / "knowledge_base" / "documents.json" )

STOP_WORDS = {
    "a", "an", "and", "are", "about", "can", "could",
    "do", "does", "did", "for", "how", "i", "in", "is",
    "create", "endpoint", "fastapi",
    "it", "me", "my", "of", "on", "or", "please", "the",
    "this", "that", "to", "what", "when", "where", "which",
    "who", "why", "with", "would", "you", "your",
}


@dataclass(frozen=True)
class Document:
    """One documentation page loaded from our local snapshot."""

    id: str
    title: str
    url: str
    keywords: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class SearchResult:
    """A document's relevance score and supporting passages."""

    document: Document
    score: int
    matched_terms: tuple[str, ...]
    passages: tuple[str, ...]


def tokenize(text: str) -> set[str]:
    """Convert text into lowercase terms, excluding common words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {
        word
        for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


def load_documents( path: Path = DEFAULT_KNOWLEDGE_BASE, ) -> list[Document]:
    """Load and validate the local documentation snapshot."""
    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("Knowledge base must be a JSON object.")

    entries = raw.get("documents")
    if not isinstance(entries, list):
        raise ValueError("Knowledge base must contain a documents list." )

    documents = []
    seen_ids = set()

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError( f"Document at index {index} must be an object." )

        for field in ("id", "title", "url", "content"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Document at index {index} has an invalid "
                    f"or missing {field!r}."
                )

        keywords = entry.get("keywords")
        if (
            not isinstance(keywords, list)
            or not all(
                isinstance(keyword, str)
                for keyword in keywords
            )
        ):
            raise ValueError( f"Document at index {index} has invalid keywords." )

        document_id = entry["id"]
        if document_id in seen_ids:
            raise ValueError(
                f"Duplicate document ID: {document_id}"
            )
        seen_ids.add(document_id)

        if not entry["url"].startswith("https://"):
            raise ValueError( f"Document {document_id} has a non-HTTPS URL." )

        documents.append(
            Document(
                id=document_id,
                title=entry["title"],
                url=entry["url"],
                keywords=tuple(keywords),
                content=entry["content"],
            )
        )

    return documents



def extract_passages(  content: str,  query_terms,  max_passages: int = 3,  max_chars: int = 1200, ) -> list[str]:
    """Return relevant text windows with surrounding context."""
    terms = set(query_terms)
    lines = [line.rstrip() for line in content.splitlines()]

    if not terms or not lines:
        return []

    candidates = []

    # Each matching line becomes the centre of a context window.
    for index, line in enumerate(lines):
        line_terms = set(tokenize(line))
        direct_matches = terms & line_terms

        if not direct_matches:
            continue

        start = max(0, index - 4)
        end = min(len(lines), index + 5)

        passage = "\n".join(lines[start:end]).strip()

        if not passage:
            continue

        passage_terms = set(tokenize(passage))
        coverage = len(terms & passage_terms)

        # Keep unusually long passages within the character limit.
        if len(passage) > max_chars:
            centre = sum(
                len(previous_line) + 1
                for previous_line in lines[start:index]
            )
            crop_start = max(0, centre - max_chars // 2)
            passage = passage[crop_start:crop_start + max_chars].strip()

        candidates.append(
            (
                coverage,
                len(direct_matches),
                start,
                end,
                passage,
            )
        )

    # Prefer passages covering more distinct query terms.
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))

    selected = []

    for coverage, direct_matches, start, end, passage in candidates:
        # Avoid returning several nearly identical context windows.
        too_similar = False

        for _, _, selected_start, selected_end, _ in selected:
            overlap = max(
                0,
                min(end, selected_end) - max(start, selected_start),
            )
            shorter_window = min(
                end - start,
                selected_end - selected_start,
            )

            if (
                shorter_window > 0
                and overlap / shorter_window >= 0.6
            ):
                too_similar = True
                break

        if too_similar:
            continue

        selected.append(
            (coverage, direct_matches, start, end, passage)
        )

        if len(selected) >= max_passages:
            break

    # Restore document order so the surrounding text remains readable.
    selected.sort(key=lambda item: item[2])

    return [item[4] for item in selected]


    

def search_documents( query: str, documents: list[Document], limit: int = 3, ) -> list[SearchResult]:
    """Rank relevant documents and reject weak lexical matches."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    query_terms = set(tokenize(query))

    if not query_terms:
        return []

    results = []

    for document in documents:
        keyword_terms = set(tokenize(" ".join(document.keywords)))
        title_terms = set(tokenize(document.title))
        content_terms = set(tokenize(document.content))

        matched_terms = query_terms & ( keyword_terms | title_terms | content_terms )

        if not matched_terms:
            continue

        # A match in metadata is stronger than a match in body text.
        score = (
            6 * len(query_terms & keyword_terms)
            + 4 * len(query_terms & title_terms)
            + len(query_terms & content_terms)
        )

        # Reject weak matches such as a single generic word.
        if score < MIN_RELEVANCE_SCORE:
            continue

        passages = extract_passages( document.content, query_terms, )

        # A title match alone is not sufficient evidence.
        if not passages:
            continue

        results.append(
            SearchResult(
                document=document,
                score=score,
                matched_terms=tuple(sorted(matched_terms)),
                passages=passages,
            )
        )

    results.sort( key=lambda result: (-result.score, result.document.id) )

    return results[:limit]
