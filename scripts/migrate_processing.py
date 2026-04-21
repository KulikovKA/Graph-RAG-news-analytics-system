import os
import sys
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Настройка путей
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.common.models import Base

# Загрузка конфигурации
load_dotenv(os.path.join(PROJECT_ROOT, '.env.db'))

# Настройки БД с обработкой отсутствующих значений
user = os.getenv('POSTGRES_USER', 'postgres')
password = os.getenv('POSTGRES_PASSWORD', 'mysecretpassword')
db_name = os.getenv('POSTGRES_DB', 'graph_rag')
host = os.getenv('POSTGRES_HOST', 'localhost').strip()
port = os.getenv('POSTGRES_PORT', '5432').strip()

DB_URL = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
engine = create_engine(DB_URL)

def migrate():
    print(f"Migrating database at {host}:{port}...")
    try:
        Base.metadata.create_all(engine)
        print("Success: ProcessingChannel table created/verified.")
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
