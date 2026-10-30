from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the project root importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from code_validation import (
    apply_code_template,
    validate_code_semantics,
    validate_python_examples,
)
from main import DOCUMENTS, app
from retrieval import search_documents

CASES_FILE = PROJECT_ROOT / "tests" / "evaluation_cases.json"


def normalize_text(text: str) -> str:
    """Normalize whitespace so line breaks do not invalidate a quote."""
    return " ".join(text.split())


def load_cases() -> list[dict]:
    """Load the evaluation cases from JSON."""
    with CASES_FILE.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("The evaluation file must contain a non-empty list.")

    return cases


def evaluate_case(client: TestClient, case: dict) -> bool:
    """Evaluate retrieval, answer behaviour, citation, and Python syntax."""
    case_id = case["id"]
    question = case["question"]
    should_answer = case["should_answer"]
    expected_url = case["expected_source_url"]
    expects_python_code = case.get("expects_python_code", False)

    print(f"\n{'=' * 70}")
    print(f"CASE: {case_id}")
    print(f"QUESTION: {question}")

    # 1. Check retrieval independently of the model-generated answer.
    search_results = search_documents(question, DOCUMENTS, limit=3)
    retrieved_urls = [result.document.url for result in search_results]

    if should_answer:
        retrieval_ok = expected_url in retrieved_urls
    else:
        retrieval_ok = len(search_results) == 0

    print(f"Retrieved URLs: {retrieved_urls or 'None'}")
    print(f"RETRIEVAL: {'PASS' if retrieval_ok else 'FAIL'}")

    # 2. Exercise the actual API endpoint.
    try:
        response = client.post("/support", json={"question": question})
    except Exception as error:
        print(f"API: FAIL — {type(error).__name__}: {error}")
        return False

    if response.status_code != 200:
        print(f"API: FAIL — HTTP {response.status_code}")
        print(response.text)
        return False

    result = response.json()
    answer = result.get("answer")
    source = result.get("source")
    
    # Check that supported concepts use the exact canonical answer.
    template_ok = True
    expectation = case.get("code_expectation")

    if should_answer and expectation is not None:
        expected_answer = apply_code_template(expectation)
        template_ok = answer == expected_answer

        print(f"ANSWER REPR: {answer!r}")
        print(
            "CANONICAL TEMPLATE: "
            f"{'PASS' if template_ok else 'FAIL'}"
        )

        if not template_ok:
            print(f"EXPECTED REPR: {expected_answer!r}")

    # 3. Check whether the agent answered or abstained as expected.
    if should_answer:
        answer_ok = (
            isinstance(answer, str)
            and bool(answer.strip())
            and result.get("evidence_sufficient") is True
        )
    else:
        answer_ok = (
            answer is None
            and source is None
            and result.get("evidence_sufficient") is False
        )

    print(f"ANSWER / ABSTENTION: {'PASS' if answer_ok else 'FAIL'}")
    print(f"ANSWER:\n{answer or '<abstained>'}")

    # 4. Check the expected source and whether the quote exists in it.
    if should_answer:
        matching_document = next(
            (doc for doc in DOCUMENTS if doc.url == expected_url),
            None,
        )

        quote = source.get("quote", "") if source else ""
        source_url = source.get("url") if source else None

        quote_exists = (
            matching_document is not None
            and bool(quote.strip())
            and normalize_text(quote)
            in normalize_text(matching_document.content)
        )

        citation_ok = source_url == expected_url and quote_exists

        print(f"SOURCE URL: {source_url or 'None'}")
        print(f"SOURCE QUOTE: {quote or 'None'}")
        print(f"CITATION: {'PASS' if citation_ok else 'FAIL'}")
    else:
        citation_ok = source is None
        print(
            "CITATION: "
            f"{'PASS' if citation_ok else 'FAIL'} "
            "(an unsupported question should have no source)"
        )

    # 5. Check syntax of expected Python code examples.
    if should_answer:
        code_ok, code_message = validate_python_examples(
            answer or "",
            expects_python_code,
            case.get("code_expectation"),
        )
        print(f"PYTHON CODE: {'PASS' if code_ok else 'FAIL'} — {code_message}")
    else:
        code_ok = True
        print("PYTHON CODE: N/A — abstention expected.")

    case_passed = (
        retrieval_ok
        and answer_ok
        and citation_ok
        and code_ok
        and template_ok
    )
    print(f"CASE RESULT: {'PASS' if case_passed else 'FAIL'}")

    return case_passed


def main() -> int:
    cases = load_cases()
    results: list[bool] = []

    print(f"Running {len(cases)} evaluation cases.")
    print("The API runs in-process; Ollama must be running.")

    with TestClient(app) as client:
        for case in cases:
            results.append(evaluate_case(client, case))

    passed = sum(results)
    failed = len(results) - passed

    print(f"\n{'=' * 70}")
    print(f"EVALUATION SUMMARY: {passed}/{len(results)} cases passed.")
    print(f"Failed cases: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
