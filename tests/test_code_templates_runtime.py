
from fastapi.testclient import TestClient

from code_validation import CODE_EXAMPLES


def create_app_from_template(expectation: str):
    """Load a trusted, repository-owned example for runtime testing."""
    namespace = {}
    exec(CODE_EXAMPLES[expectation], namespace)
    return namespace["app"]


def test_get_endpoint_template_runs():
    app = create_app_from_template("get_endpoint")

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_path_parameter_template_runs():
    app = create_app_from_template("path_parameter")

    with TestClient(app) as client:
        response = client.get("/items/123")

    assert response.status_code == 200
    assert response.json() == {"item_id": 123}


def test_query_parameter_template_runs():
    app = create_app_from_template("query_parameter")

    with TestClient(app) as client:
        response = client.get("/items/?needy=coffee")

    assert response.status_code == 200
    assert response.json() == {"needy": "coffee"}


def test_query_parameter_is_required():
    app = create_app_from_template("query_parameter")

    with TestClient(app) as client:
        response = client.get("/items/")

    assert response.status_code == 422


def test_request_body_template_runs():
    app = create_app_from_template("request_body")

    with TestClient(app) as client:
        response = client.post(
            "/items/",
            json={
                "name": "Coffee",
                "price": 3.5,
            },
        )

    assert response.status_code == 200
    assert response.json()["name"] == "Coffee"
    assert response.json()["price"] == 3.5


def test_request_body_rejects_missing_required_field():
    app = create_app_from_template("request_body")

    with TestClient(app) as client:
        response = client.post(
            "/items/",
            json={"name": "Coffee"},
        )

    assert response.status_code == 422
