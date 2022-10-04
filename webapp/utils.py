from classifier import BirdClassifier
from webapp.model import Batch, BatchResult, SessionLocal


def run_batch(batch_id: str):
    db = SessionLocal()
    batch = db.query(Batch).filter(Batch.batch_id == batch_id).one_or_none()
    if not batch:
        return
    for index, url in enumerate(batch.urls):
        from webapp.webapp import q

        q.enqueue(run_classifier, url, batch_id, index)


def run_classifier(url, batch_id, index):
    classifier = BirdClassifier()
    result = {}
    bird_classification = classifier.main(url)
    top_classification = classifier.show_json_response(bird_classification)
    result = BatchResult(
        batch_id=batch_id, result=top_classification, index=index + 1, url=url
    )
    db = SessionLocal()
    db.add(result)
    db.commit()
    batch = db.query(Batch).filter(Batch.batch_id == batch_id).one_or_none()
    result_count = (
        db.query(BatchResult).filter(BatchResult.batch_id == batch_id).count()
    )
    if result_count == batch.total_urls:
        batch.status = "completed"
        db.commit()
    return
