import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, '.env.db'))

user = os.getenv('POSTGRES_USER', 'postgres')
password = os.getenv('POSTGRES_PASSWORD', 'mysecretpassword')
db_name = os.getenv('POSTGRES_DB', 'graph_rag')
host = os.getenv('POSTGRES_HOST', 'localhost').strip()
port = os.getenv('POSTGRES_PORT', '5432').strip()

DB_URL = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

from src.common.models import ProcessingChannel

def check():
    session = SessionLocal()
    channels = session.query(ProcessingChannel).all()
    print("--- Содержимое таблицы processing_channels ---")
    if not channels:
        print("Таблица пуста.")
    for c in channels:
        print(f"ID: {c.id} | Username: @{c.username} | Status: {c.status} | Progress: {c.progress}%")
    print("---------------------------------------------")

if __name__ == "__main__":
    try:
        check()
    except Exception as e:
        print(f"Ошибка при подключении к БД: {e}")
