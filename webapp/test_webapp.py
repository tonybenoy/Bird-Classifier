from fastapi.testclient import TestClient

from webapp.webapp import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "It works"}


def test_get_match():
    response = client.get(
        "/url/?url=https://www.allaboutbirds.org/guide/PHOTO/LARGE/blue_jay_8.jpg"
    )
    assert response.status_code == 200
    assert "second_match" in response.json().keys()


# Don't have test cases for the rest of the endpoints because
# they require a database and I would  need to setup fixtures for that.
