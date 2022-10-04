import uuid
from typing import List

from fastapi import Depends, FastAPI
from redis import Redis
from rq import Queue
from sqlalchemy.orm import Session

from classifier import BirdClassifier
from webapp.model import Batch, BatchResult, get_db
from webapp.settings import Settings
from webapp.utils import run_batch

app = FastAPI()

redis_conn = Redis(host=Settings.REDIS_HOST, port=Settings.REDIS_PORT)
q = Queue("default", connection=redis_conn)


@app.get("/")
async def root():
    return {"message": "It works"}


@app.get("/url/")
async def get_match(url: str):
    classifier = BirdClassifier()
    result = {}
    bird_classification = classifier.main(url)
    top_classification = classifier.show_json_response(bird_classification)
    result.update(top_classification)
    return result


@app.post("/batch/")
async def create_batch(urls: List[str], db: Session = Depends(get_db)):
    batch_id = str(uuid.uuid4())
    batch = Batch(batch_id=batch_id, urls=urls, status="pending", total_urls=len(urls))
    db.add(batch)
    db.commit()
    q.enqueue(run_batch, batch_id)
    return {"batch_id": batch_id}


@app.get("/batch/{batch_id}")
async def get_batch(batch_id: str, db: Session = Depends(get_db)):
    batch = db.query(Batch).filter(Batch.batch_id == batch_id).first()
    if batch:
        if batch.status == "pending":
            return {"status": "pending", "batch_id": batch_id}
        elif batch.status == "completed":
            results = (
                db.query(BatchResult).filter(BatchResult.batch_id == batch_id).all()
            )
            response = {}
            for result in results:
                response[result.index] = result.result
            return {
                "status": "completed",
                "batch_id": batch_id,
                "results": response,
            }
        elif batch.status == "failed":
            return {"status": "failed", "batch_id": batch_id}
    else:
        return {"status": "not found"}


@app.get("/batch/{batch_id}/cancel")
async def cancel_batch(batch_id: str, db: Session = Depends(get_db)):
    batch = db.query(Batch).filter(Batch.batch_id == batch_id).first()
    if batch:
        if batch.status == "pending":
            batch.status = "cancelled"
            db.commit()
            return {"status": "cancelled", "batch_id": batch_id}
        elif batch.status == "completed":
            return {"status": "completed", "batch_id": batch_id}
        elif batch.status == "failed":
            return {"status": "failed", "batch_id": batch_id}
    else:
        return {"status": "not found"}
