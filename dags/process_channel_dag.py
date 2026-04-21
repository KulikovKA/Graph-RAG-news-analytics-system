import os
import sys
import asyncio
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.models import Param
from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker

# Добавляем корень проекта в пути
PROJECT_ROOT = "/opt/airflow" # Стандартный путь в Docker
sys.path.append(PROJECT_ROOT)

from src.common.models import ProcessingChannel, News
from src.scraper.tg_api_scraper import TelegramClient, scrape_channel
from src.processor.extraction_service import graph

# Настройки БД
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'mysecretpassword')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'graph_rag')

DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

def update_db_status(username, status, progress, error_message=None):
    with SessionLocal() as session:
        values = {"status": status, "progress": progress}
        if error_message:
            values["error_message"] = error_message
        session.execute(
            update(ProcessingChannel)
            .where(ProcessingChannel.username == username)
            .values(**values)
        )
        session.commit()

def on_failure_callback(context):
    params = context.get('params', {})
    username = params.get('channel_username')
    if username:
        error = str(context.get('exception', 'Unknown Airflow error'))
        update_db_status(username, "ERROR", 0, error_message=error)

@dag(
    dag_id="process_new_channel",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    on_failure_callback=on_failure_callback,
    params={
        "channel_username": Param(type="string", description="Username of the channel"),
        "channel_url": Param(type="string", description="Full URL of the channel")
    },
    tags=["graphrag", "ingestion"]
)
def process_channel_dag():
    
    @task()
    def scrape_step(**kwargs):
        username = kwargs['params']['channel_username']
        url = kwargs['params']['channel_url']
        
        update_db_status(username, "SCRAPING", 5)
        
        # Настройки API
        API_ID = int(os.getenv("TG_API_ID"))
        API_HASH = os.getenv("TG_API_HASH")
        SESSION_PATH = os.path.join(PROJECT_ROOT, "anon.session")
        
        async def run_scraping():
            client = TelegramClient(SESSION_PATH, API_ID, API_HASH)
            await client.start()
            
            update_db_status(username, "SCRAPING", 15)
            
            # Лимит для нового канала
            NEW_CHANNEL_LIMIT = int(os.getenv("NEW_CHANNEL_LIMIT", 200))
            
            with SessionLocal() as db_session:
                await scrape_channel(client, db_session, url, limit=NEW_CHANNEL_LIMIT)
            
            update_db_status(username, "SCRAPING", 40)
            await client.disconnect()

        asyncio.run(run_scraping())
        update_db_status(username, "SCRAPING", 50)

    @task()
    def extraction_step(**kwargs):
        username = kwargs['params']['channel_username']
        update_db_status(username, "EXTRACTING", 55)
        
        async def run_extraction():
            with SessionLocal() as session:
                update_db_status(username, "EXTRACTING", 60)
                # Берем все необработанные новости для этого канала
                from sqlalchemy import select
                query = select(News).where(News.source == username.lower(), News.is_processed == False)
                batch = list(session.execute(query).scalars().all())
                
                if batch:
                    print(f"Extracting entities for {len(batch)} messages...")
                    update_db_status(username, "EXTRACTING", 70)
                    # Запускаем наш LangGraph процессор
                    await graph.ainvoke({"batch": batch, "channel_username": username})
                    update_db_status(username, "EXTRACTING", 95)
                else:
                    print("No messages found for extraction.")

        asyncio.run(run_extraction())
        update_db_status(username, "EXTRACTING", 98)

    @task()
    def finalize_step(**kwargs):
        username = kwargs['params']['channel_username']
        update_db_status(username, "ACTIVE", 100)
        print(f"Channel @{username} is now ACTIVE.")

    # Определение цепочки
    scrape_step() >> extraction_step() >> finalize_step()

process_channel_dag()
