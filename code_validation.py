import ast
import re


PYTHON_BLOCK_PATTERN = re.compile(
    r"```python\s*\n(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)


CODE_EXAMPLES = {
    "get_endpoint": '''from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}''',

    "path_parameter": '''from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}''',

    "query_parameter": '''from fastapi import FastAPI

app = FastAPI()

@app.get("/items/")
async def read_items(needy: str):
    return {"needy": needy}''',

    "optional_query_parameter": '''from fastapi import FastAPI

app = FastAPI()

@app.get("/items/")
async def read_items(q: str | None = None):
    return {"q": q}''',

    "request_body": '''from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float


@app.post("/items/")
async def create_item(item: Item):
    return item''',
}


CODE_EXPLANATIONS = {
    "get_endpoint": (
        "A basic GET endpoint uses a route decorator to associate a URL "
        "with a function. This example registers the root path and returns "
        "a dictionary as the response."
    ),
    "path_parameter": (
        "A path parameter is declared in the route using braces around "
        "its name. The handler function declares a parameter with the "
        "same name and a type annotation."
    ),
    "query_parameter": (
        "FastAPI treats a function parameter that is not part of the "
        "route path as a query parameter. In this example, needy is "
        "required because it has no default value."
    ),
    "optional_query_parameter": (
        "An optional query parameter has a default value. "
        "Here, q accepts a string or None. "
        "FastAPI uses None when the client omits it."
    ),
    "request_body": (
        "To receive a JSON request body, define a Pydantic model and "
        "use it as a parameter annotation in the endpoint function. "
        "FastAPI interprets it as a request body and validates the data."
    ),
}


EXPECTED_SOURCE_URLS = {
    "get_endpoint": (
        "https://fastapi.tiangolo.com/tutorial/first-steps/"
    ),
    "path_parameter": (
        "https://fastapi.tiangolo.com/tutorial/path-params/"
    ),
    "query_parameter": (
        "https://fastapi.tiangolo.com/tutorial/query-params/"
    ),
    "optional_query_parameter": (
        "https://fastapi.tiangolo.com/tutorial/query-params/"
    ),
    "request_body": (
        "https://fastapi.tiangolo.com/tutorial/body/"
    ),
}


def expected_source_url(expectation: str | None) -> str | None:
    """Return the official documentation URL expected for a known concept."""
    if expectation is None:
        return None

    return EXPECTED_SOURCE_URLS.get(expectation)



def select_evidence_quote(
    passages: list[str],
    expectation: str | None,
) -> str | None:
    """Select a quote from retrieved evidence relevant to the expected topic."""
    if expectation is None:
        return None

    keywords = {
        "get_endpoint": (
            "@app.get",
            "get endpoint",
            "path operation",
        ),
        "path_parameter": (
            "path parameter",
            "path parameters",
        ),
        "query_parameter": (
            "query parameter",
            "query parameters",
        ),
        "optional_query_parameter": (
            "optional query parameter",
            "default value",
            "optional",
        ),
        "request_body": (
            "request body",
            "pydantic model",
        ),
    }

    topic_keywords = keywords.get(expectation)

    if topic_keywords is None:
        return None

    # Prefer a concise, relevant line from the retrieved passages.
    for passage in passages:
        for line in passage.splitlines():
            candidate = " ".join(line.split())

            if len(candidate) >= 35 and any(
                keyword in candidate.casefold()
                for keyword in topic_keywords
            ):
                return candidate

    # If no single line is sufficiently informative, use a relevant passage.
    for passage in passages:
        candidate = " ".join(passage.split())

        if candidate and any(
            keyword in candidate.casefold()
            for keyword in topic_keywords
        ):
            return candidate

    return None



def validate_code_semantics(
    tree: ast.Module,
    expectation: str,
) -> str | None:
    """Check whether Python code demonstrates the expected FastAPI concept."""

    handlers = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        arguments = (
            node.args.posonlyargs
            + node.args.args
            + node.args.kwonlyargs
        )

        parameter_names = {argument.arg for argument in arguments}
        annotated_parameter_names = {
            argument.arg
            for argument in arguments
            if argument.annotation is not None
        }

        annotation_names = {
            argument.annotation.id
            for argument in arguments
            if isinstance(argument.annotation, ast.Name)
        }
        
                # Identify function parameters that have default values.
        positional_arguments = (
            node.args.posonlyargs + node.args.args
        )

        defaulted_parameter_names = set()

        if node.args.defaults:
            defaulted_parameter_names.update(
                argument.arg
                for argument in positional_arguments[
                    -len(node.args.defaults):
                ]
            )

        defaulted_parameter_names.update(
            argument.arg
            for argument, default in zip(
                node.args.kwonlyargs,
                node.args.kw_defaults,
            )
            if default is not None
        )

        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue

            if not isinstance(decorator.func, ast.Attribute):
                continue

            method = decorator.func.attr

            if method not in {"get", "post"}:
                continue

            if not decorator.args:
                continue

            route_argument = decorator.args[0]

            if not (
                isinstance(route_argument, ast.Constant)
                and isinstance(route_argument.value, str)
            ):
                continue

            route_path = route_argument.value

            path_parameters = set(
                re.findall(
                    r"\{([A-Za-z_]\w*)(?::[^}]*)?\}",
                    route_path,
                )
            )

            handlers.append(
                {
                    "method": method,
                    "path_parameters": path_parameters,
                    "parameter_names": parameter_names,
                    "annotation_names": annotation_names,
                    "annotated_parameter_names": annotated_parameter_names,
                    "defaulted_parameter_names": (
                        defaulted_parameter_names
                    ),
                }
            )

    if expectation == "get_endpoint":
        if any(handler["method"] == "get" for handler in handlers):
            return None

        return "No GET route using a FastAPI route decorator was found."

    if expectation == "path_parameter":
        for handler in handlers:
            if handler["method"] != "get":
                continue

            path_parameters = handler["path_parameters"]
            declared_parameters = handler["parameter_names"]
            annotated_parameters = handler["annotated_parameter_names"]

            if (
                path_parameters
                and path_parameters.issubset(declared_parameters)
                and path_parameters.issubset(annotated_parameters)
            ):
                return None

        return (
            "No GET route was found where a path parameter in the URL "
            "is also declared in the handler signature."
        )

    if expectation == "query_parameter":
        for handler in handlers:
            if handler["method"] != "get":
                continue

            # Keep this example focused on query parameters.
            if handler["path_parameters"]:
                continue

            query_parameters = (
                handler["parameter_names"]
                - {"self", "cls", "request"}
            )

            if query_parameters:
                return None

        return (
            "No simple GET route was found with a query parameter "
            "and no path parameters."
        )


    if expectation == "optional_query_parameter":
        for handler in handlers:
            if handler["method"] != "get":
                continue

            # Keep this example focused on query parameters.
            if handler["path_parameters"]:
                continue

            query_parameters = (
                handler["parameter_names"]
                - {"self", "cls", "request"}
            )

            optional_parameters = (
                query_parameters
                & handler["defaulted_parameter_names"]
            )

            if optional_parameters:
                return None

        return (
            "No simple GET route was found with an optional "
            "query parameter that has a default value."
        )


    if expectation == "request_body":
        model_names = set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            inherits_base_model = any(
                isinstance(base, ast.Name)
                and base.id == "BaseModel"
                for base in node.bases
            )

            if inherits_base_model:
                model_names.add(node.name)

        for handler in handlers:
            if handler["method"] != "post":
                continue

            if handler["annotation_names"] & model_names:
                return None

        return (
            "No POST route was found with a handler parameter "
            "annotated using a locally defined Pydantic BaseModel."
        )

    return f"Unknown code expectation: {expectation}"




def _has_imported_name(
    tree: ast.Module,
    module: str,
    name: str,
) -> bool:
    """Check whether a module imports a specific name."""

    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue

        if node.module != module:
            continue

        for imported_name in node.names:
            if imported_name.name != name:
                continue

            if imported_name.asname is None or imported_name.asname == name:
                return True

    return False

def validate_python_examples(
    answer: str,
    required: bool,
    expectation: str | None = None,
) -> tuple[bool, str]:
    """Check Python formatting, syntax, and the expected code structure."""

    code_blocks = PYTHON_BLOCK_PATTERN.findall(answer)
    
    if not code_blocks:
        if required:
            return (
                False,
                "Expected a fenced ```python code block, but none was returned.",
            )

        return True, "No Python code example required."



    parsed_trees = []

    for block_number, code in enumerate(code_blocks, start=1):
        try:
            tree = ast.parse(code)
        except SyntaxError as error:
            return (
                False,
                f"Python code block {block_number} has a syntax error "
                f"at line {error.lineno}: {error.msg}",
            )

        parsed_trees.append(tree)

    if expectation is not None:
        validation_errors = []

        for block_number, tree in enumerate(parsed_trees, start=1):
            semantic_error = validate_code_semantics(
                tree,
                expectation,
            )

            if semantic_error is not None:
                validation_errors.append(semantic_error)
                continue

            if not _has_imported_name(tree, "fastapi", "FastAPI"):
                validation_errors.append(
                    f"Python code block {block_number} must include "
                    "'from fastapi import FastAPI'."
                )
                continue

            if (
                expectation == "request_body"
                and not _has_imported_name(tree, "pydantic", "BaseModel")
            ):
                validation_errors.append(
                    f"Python code block {block_number} must include "
                    "'from pydantic import BaseModel'."
                )
                continue

            # At least one complete code block passed every check.
            return (
                True,
                f"At least one Python code block passed syntax validation, "
                f"the '{expectation}' structural check, and required import checks.",
            )

        return (
            False,
            f"No Python code block met the '{expectation}' expectation. "
            f"First issue: {validation_errors[0]}",
        )

    return (
        True,
        f"{len(code_blocks)} Python code block(s) passed syntax validation"
        + (
            f" and the '{expectation}' structural check."
            if expectation is not None
            else "."
        ),
    )


def apply_code_template(expectation: str | None, ) -> str | None:
    """Return a complete, predefined answer for a supported concept."""
    if expectation is None:
        return None
    code = CODE_EXAMPLES.get(expectation)
    explanation = CODE_EXPLANATIONS.get(expectation)
    if code is None or explanation is None:
        return None
    code_block = f"```python\n{code}\n```"
    return f"{explanation}\n\n{code_block}"



def find_python_syntax_error(
    answer: str,
    *,
    require_code_block: bool = False,
) -> str | None:
    """Return a syntax error for a fenced Python example, if one exists."""
    code_blocks = PYTHON_BLOCK_PATTERN.findall(answer)

    if not code_blocks:
        if require_code_block:
            return "Expected a fenced ```python code block, but none was returned."
        return None

    for block_number, code in enumerate(code_blocks, start=1):
        try:
            ast.parse(code)
        except SyntaxError as error:
            return (
                f"Python code block {block_number} has a syntax error "
                f"at line {error.lineno}: {error.msg}"
            )

    return None



def infer_code_expectation(question: str) -> str | None:
    """Identify the FastAPI concept the code example should demonstrate."""
    question = " ".join(question.casefold().split())

    is_query_parameter = (
        "query parameter" in question
        or "query param" in question
    )

    if is_query_parameter:
        optional_markers = (
            "optional",
            "not required",
            "can be omitted",
            "default value",
        )

        if any(marker in question for marker in optional_markers):
            return "optional_query_parameter"

        return "query_parameter"

    if "path parameter" in question or "path param" in question:
        return "path_parameter"

    if "request body" in question or "json body" in question:
        return "request_body"

    if "get endpoint" in question or "get route" in question:
        return "get_endpoint"

    return None