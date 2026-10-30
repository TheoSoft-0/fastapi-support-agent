
import ast

from code_validation import validate_code_semantics


def test_get_endpoint_expectation_passes():
    code = '''
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
'''

    assert validate_code_semantics(
        ast.parse(code),
        "get_endpoint",
    ) is None


def test_path_parameter_must_be_declared_by_handler():
    valid_code = '''
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
'''

    invalid_code = '''
@app.get("/items/{item_id}")
async def read_item():
    return {"message": "item"}
'''

    assert validate_code_semantics(
        ast.parse(valid_code),
        "path_parameter",
    ) is None

    assert validate_code_semantics(
        ast.parse(invalid_code),
        "path_parameter",
    ) is not None


def test_query_example_does_not_need_a_path_parameter():
    code = '''
@app.get("/items")
async def list_items(skip: int = 0):
    return {"skip": skip}
'''

    assert validate_code_semantics(
        ast.parse(code),
        "query_parameter",
    ) is None


def test_query_example_with_path_parameter_is_rejected():
    code = '''
@app.get("/items/{item_id}")
async def read_items(item_id: int, skip: int = 0):
    return {"item_id": item_id, "skip": skip}
'''

    assert validate_code_semantics(
        ast.parse(code),
        "query_parameter",
    ) is not None


def test_request_body_requires_pydantic_model_parameter():
    valid_code = '''
class Item(BaseModel):
    name: str

@app.post("/items")
async def create_item(item: Item):
    return item
'''

    invalid_code = '''
@app.post("/items")
async def create_item(item: dict):
    return item
'''

    assert validate_code_semantics(
        ast.parse(valid_code),
        "request_body",
    ) is None

    assert validate_code_semantics(
        ast.parse(invalid_code),
        "request_body",
    ) is not None
