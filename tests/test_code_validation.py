import textwrap

from code_validation import (
    CODE_EXPLANATIONS,
    apply_code_template,
    expected_source_url,
    find_python_syntax_error,
    infer_code_expectation,
    select_evidence_quote,
    validate_python_examples,
)


def test_supported_concepts_map_to_expected_documentation():
    assert expected_source_url("get_endpoint") == (
        "https://fastapi.tiangolo.com/tutorial/first-steps/"
    )
    assert expected_source_url("path_parameter") == (
        "https://fastapi.tiangolo.com/tutorial/path-params/"
    )
    assert expected_source_url("query_parameter") == (
        "https://fastapi.tiangolo.com/tutorial/query-params/"
    )
    assert expected_source_url("optional_query_parameter") == (
        "https://fastapi.tiangolo.com/tutorial/query-params/"
    )
    assert expected_source_url("request_body") == (
        "https://fastapi.tiangolo.com/tutorial/body/"
    )

    assert expected_source_url(None) is None
    assert expected_source_url("unknown_concept") is None


def test_get_endpoint_explanation_has_correct_spacing():
    explanation = CODE_EXPLANATIONS["get_endpoint"]

    assert "returns a dictionary" in explanation
    assert "returnsa" not in explanation


def test_valid_python_code_passes_syntax_validation():
    answer = textwrap.dedent("""\
        ```python
        from fastapi import FastAPI

        app = FastAPI()

        @app.post("/items")
        async def create_item():
            return {"message": "created"}
        ```
    """)

    assert find_python_syntax_error(answer) is None


def test_invalid_route_decorator_is_detected():
    answer = textwrap.dedent("""\
        ```python
        from fastapi import FastAPI

        app = FastAPI()

        @app.post="/items"
        async def create_item():
            return {"message": "created"}
        ```
    """)

    error = find_python_syntax_error(answer)

    assert error is not None
    assert "syntax error" in error


def test_missing_code_block_is_detected_when_required():
    answer = "Here is how to create a route."

    error = find_python_syntax_error(
        answer,
        require_code_block=True,
    )

    assert error is not None
    assert "fenced" in error


def test_all_code_templates_pass_structural_validation():
    expectations = [
        "get_endpoint",
        "path_parameter",
        "query_parameter",
        "optional_query_parameter",
        "request_body",
    ]

    for expectation in expectations:
        answer = apply_code_template(expectation)

        assert answer is not None
        assert "```python\n" in answer

        valid, message = validate_python_examples(
            answer,
            required=True,
            expectation=expectation,
        )

        assert valid, f"{expectation}: {message}"


def test_unknown_expectation_has_no_template():
    assert apply_code_template("unknown_concept") is None


def test_question_classification_distinguishes_optional_query_parameters():
    assert (
        infer_code_expectation(
            "How do I make a query parameter optional in FastAPI?"
        )
        == "optional_query_parameter"
    )

    assert (
        infer_code_expectation(
            "How do I read a query parameter in FastAPI?"
        )
        == "query_parameter"
    )


def test_optional_query_parameter_template_has_default_value():
    answer = apply_code_template("optional_query_parameter")

    assert answer is not None
    assert "q: str | None = None" in answer


def test_optional_query_parameter_passes_semantic_validation():
    answer = apply_code_template("optional_query_parameter")

    assert answer is not None

    valid, message = validate_python_examples(
        answer,
        required=True,
        expectation="optional_query_parameter",
    )

    assert valid, message


def test_required_query_parameter_fails_optional_validation():
    answer = apply_code_template("query_parameter")

    assert answer is not None

    valid, message = validate_python_examples(
        answer,
        required=True,
        expectation="optional_query_parameter",
    )

    assert not valid
    assert "optional_query_parameter" in message


def test_optional_query_parameter_selects_relevant_evidence():
    passages = [
        "An optional query parameter can have a default value."
    ]

    quote = select_evidence_quote(
        passages,
        "optional_query_parameter",
    )

    assert quote == (
        "An optional query parameter can have a default value."
    )