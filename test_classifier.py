from classifier import BirdClassifier
from constants import DEFAULT_IMAGE_URLS


def test_main():
    classifier = BirdClassifier()
    resp = classifier.main(DEFAULT_IMAGE_URLS[1])
    assert "name" in resp[0][1].keys()


def test_main_invalid_url():
    classifier = BirdClassifier(model_url="test_data/bird_model")
    resp = classifier.main("sss")
    assert resp == []


def test_main_local_model():
    classifier = BirdClassifier(model_url="test_data/bird_model")
    resp = classifier.main(DEFAULT_IMAGE_URLS[1])
    assert "name" in resp[0][1].keys()


def test_main_local_model_local_csv():
    classifier = BirdClassifier(
        model_url="test_data/bird_model",
        labels_url="test_data/csv/aiy_birds_V1_labelmap.csv",
    )
    resp = classifier.main(DEFAULT_IMAGE_URLS[1])
    assert "name" in resp[0][1].keys()
