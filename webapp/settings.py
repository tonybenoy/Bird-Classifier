import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT")) or 5432
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "python-test")
    DATABASE_URL: str = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@\
        {POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT")) or 6379
