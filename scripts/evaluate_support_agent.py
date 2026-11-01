from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

# Make the project root importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

import main as app_module
from code_validation import (
    apply_code_template,
    validate_python_examples,
)
from main import DOCUMENTS, app
from retrieval import search_documents


@dataclass
class CaseTrace:
    """Record model-generation and recovery behaviour for one case."""

    expects_python_code: bool
    code_expectation: str | None

    initial_draft_returned: bool = False
    initial_answerable: bool | None = None
    initial_code_valid: bool | None = None
    initial_answer: str | None = None
    initial_validation_error: str | None = None


    fallback_attempted: bool = False
    fallback_used: bool = False

    correction_attempted: bool = False
    corrected_code_valid: bool | None = None
    corrected_answer: str | None = None
    corrected_validation_error: str | None = None


@dataclass
class EvaluationTrace:
    """Collect measurements across evaluation cases."""

    current_case: CaseTrace | None = None
    cases: list[CaseTrace] = field(default_factory=list)


@contextmanager
def instrument_application(trace: EvaluationTrace):
    """Measure model generation and recovery without changing application logic."""

    original_support_run = app_module.support_agent.run
    original_correction_run = app_module.correction_agent.run
    original_apply_code_template = app_module.apply_code_template

    async def tracked_support_run(*args, **kwargs):
        result = await original_support_run(*args, **kwargs)

        current_case = trace.current_case

        if current_case is not None and current_case.expects_python_code:
            draft = result.output

            current_case.initial_draft_returned = True
            current_case.initial_answerable = draft.can_answer
            current_case.initial_answer = draft.answer

            if draft.can_answer:
                code_ok, code_message = validate_python_examples(
                    draft.answer or "",
                    required=True,
                    expectation=current_case.code_expectation,
                )

                current_case.initial_code_valid = code_ok

                current_case.initial_validation_error = (
                    None if code_ok else code_message
                )

        return result
    async def tracked_correction_run(*args, **kwargs):
        case = trace.current_case

        if case is not None:
            case.correction_attempted = True

        try:
            result = await original_correction_run(*args, **kwargs)

        except Exception as error:
            if case is not None:
                case.corrected_code_valid = False
                case.corrected_validation_error = ( f"{type(error).__name__}: {error}" )

            raise

        if case is not None and case.expects_python_code:
            draft = result.output
            case.corrected_answer = draft.answer

            if not draft.can_answer:
                case.corrected_code_valid = False
                case.corrected_validation_error = (
                    draft.reason_if_not_answerable
                    or "The correction agent marked the answer unsupported."
                )

            elif not draft.answer or not draft.answer.strip():
                case.corrected_code_valid = False
                case.corrected_validation_error = ( "The corrected draft contained no usable answer." )

            else:
                code_valid, code_message = validate_python_examples(
                    draft.answer,
                    required=True,
                    expectation=case.code_expectation,
                )

                case.corrected_code_valid = code_valid
                case.corrected_validation_error = ( None if code_valid else code_message )

        return result



    def tracked_apply_code_template(expectation):
        case = trace.current_case

        if case is not None:
            case.fallback_attempted = True

        result = original_apply_code_template(expectation)

        if case is not None and result is not None:
            case.fallback_used = True

        return result

    with (
        patch.object(
            app_module.support_agent,
            "run",
            new=tracked_support_run,
        ),
        patch.object(
            app_module.correction_agent,
            "run",
            new=tracked_correction_run,
        ),
        patch.object(
            app_module,
            "apply_code_template",
            new=tracked_apply_code_template,
        ),
    ):
        yield


def print_case_diagnostics(trace: CaseTrace) -> None:
    """Print the model-generation and recovery results for one case."""

    if not trace.expects_python_code:
        return

    def result_label(value: bool | None) -> str:
        if value is None:
            return "N/A"

        return "PASS" if value else "FAIL"

    print(
        "INITIAL MODEL DRAFT: "
        f"{'RETURNED' if trace.initial_draft_returned else 'NOT RETURNED'}"
    )

    print(f"INITIAL ANSWERABLE: {trace.initial_answerable}")

    print(
        "INITIAL CODE VALIDATION: "
        f"{result_label(trace.initial_code_valid)}"
    )

    print(
        "FALLBACK: "
        f"attempted={trace.fallback_attempted}, "
        f"used={trace.fallback_used}"
    )

    print(f"CORRECTION ATTEMPTED: {trace.correction_attempted}")

    print("INITIAL MODEL ANSWER:", repr(trace.initial_answer))
    print("INITIAL VALIDATION ERROR:", trace.initial_validation_error)

    if trace.correction_attempted:
        print("CORRECTED MODEL ANSWER:", repr(trace.corrected_answer))
        print("CORRECTED VALIDATION ERROR:", trace.corrected_validation_error)
        print(
            "CORRECTED CODE VALIDATION: "
            f"{result_label(trace.corrected_code_valid)}"
        )

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


def evaluate_case(
    client: TestClient,
    case: dict,
    trace: EvaluationTrace,
) -> bool:
    """Evaluate retrieval, answer behaviour, citation, and Python syntax."""
    case_id = case["id"]
    question = case["question"]
    should_answer = case["should_answer"]
    expected_url = case["expected_source_url"]
    expects_python_code = case.get("expects_python_code", False)
    case_trace = CaseTrace(
        expects_python_code=expects_python_code,
        code_expectation=case.get("code_expectation"),
    )

    trace.cases.append(case_trace)
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
    trace.current_case = case_trace

    try:
        response = client.post(
            "/support",
            json={"question": question},
        )
    except Exception as error:
        print(f"API: FAIL — {type(error).__name__}: {error}")
        return False
    finally:
        trace.current_case = None

    print_case_diagnostics(case_trace)

    if response.status_code != 200:
        print(f"API: FAIL — HTTP {response.status_code}")
        print(response.text)
        return False

    result = response.json()
    answer = result.get("answer")
    source = result.get("source")

    # Check canonical templates only when the fallback was used.
    template_ok = True
    expectation = case.get("code_expectation")

    if should_answer and expectation is not None:
        print(f"ANSWER REPR: {answer!r}")

        if case_trace.fallback_used:
            expected_answer = apply_code_template(expectation)
            template_ok = answer == expected_answer

            print(
                "CANONICAL TEMPLATE: "
                f"{'PASS' if template_ok else 'FAIL'}"
            )

            if not template_ok:
                print(f"EXPECTED REPR: {expected_answer!r}")
        else:
            print("CANONICAL TEMPLATE: N/A — fallback was not used.")

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


def print_generation_summary(trace: EvaluationTrace) -> None:
    """Print aggregate model-generation and recovery measurements."""

    code_cases = [
        case
        for case in trace.cases
        if case.expects_python_code
    ]

    if not code_cases:
        return

    total = len(code_cases)

    drafts_returned = sum(
        case.initial_draft_returned
        for case in code_cases
    )

    initial_code_valid = sum(
        case.initial_code_valid is True
        for case in code_cases
    )

    fallback_used = sum(
        case.fallback_used
        for case in code_cases
    )

    correction_attempts = sum(
        case.correction_attempted
        for case in code_cases
    )

    correction_successes = sum(
        case.corrected_code_valid is True
        for case in code_cases
        if case.correction_attempted
    )

    print(f"\n{'=' * 70}")
    print("MODEL GENERATION AND RECOVERY METRICS")

    print(f"Initial drafts returned: {drafts_returned}/{total}")
    print(f"Initial code validation passed: {initial_code_valid}/{total}")
    print(f"Fallback templates used: {fallback_used}/{total}")
    print(f"Correction attempts: {correction_attempts}/{total}")

    if correction_attempts:
        print(
            "Corrected code validation passed: "
            f"{correction_successes}/{correction_attempts}"
        )
    else:
        print(
            "Corrected code validation: N/A — "
            "the correction agent was never called."
        )


def main() -> int:
    cases = load_cases()
    results: list[bool] = []

    print(f"Running {len(cases)} evaluation cases.")
    print("The API runs in-process; Ollama must be running.")

    trace = EvaluationTrace()

    with instrument_application(trace):
        with TestClient(app) as client:
            for case in cases:
                results.append(
                    evaluate_case(client, case, trace)
                )

    print_generation_summary(trace)

    passed = sum(results)
    failed = len(results) - passed

    print(f"\n{'=' * 70}")
    print(f"EVALUATION SUMMARY: {passed}/{len(results)} cases passed.")
    print(f"Failed cases: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
