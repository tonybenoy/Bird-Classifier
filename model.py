import datetime

from sqlalchemy import (
    ARRAY,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "postgresql://postgres:postgres@localhost/python-test"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class BaseMixin(Base):
    __abstract__ = True

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )


class Batch(BaseMixin):
    __tablename__ = "batch"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String(50), unique=True, index=True)
    urls = Column(ARRAY(String(1000)))
    status = Column(String(50))
    total_urls = Column(Integer)


class BatchResult(BaseMixin):
    __tablename__ = "batch_result"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(
        String(50), ForeignKey(Batch.batch_id), index=True, nullable=False
    )
    index = Column(Integer)
    url = Column(String(1000), nullable=False)
    result = Column(JSON, nullable=False, default={})


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
