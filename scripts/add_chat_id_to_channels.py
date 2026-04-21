import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Настройка путей
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, '.env.db'))

user = os.getenv('POSTGRES_USER', 'postgres')
password = os.getenv('POSTGRES_PASSWORD', 'mysecretpassword')
db_name = os.getenv('POSTGRES_DB', 'graph_rag')
host = os.getenv('POSTGRES_HOST', 'localhost')
port = os.getenv('POSTGRES_PORT', '5432')

DB_URL = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
engine = create_engine(DB_URL)

def migrate():
    with engine.connect() as conn:
        print("Migrating processing_channels table...")
        
        # Используем современный синтаксис ADD COLUMN IF NOT EXISTS
        conn.execute(text("ALTER TABLE processing_channels ADD COLUMN IF NOT EXISTS last_chat_id BIGINT"))
        print("last_chat_id column checked/added.")
        
        conn.execute(text("ALTER TABLE processing_channels ADD COLUMN IF NOT EXISTS last_message_id BIGINT"))
        print("last_message_id column checked/added.")
        
        conn.commit()
        print("Migration completed successfully.")

if __name__ == "__main__":
    migrate()
