import os
import sys
import asyncio
import requests
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from sqlalchemy import create_engine, update, select
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

# Настройки Telegram уведомлений
ADMIN_CHAT_ID = 2105841445
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")

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

@dag(
    dag_id="daily_channels_update",
    schedule="30 3 * * *", # Ежедневно в 03:30
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["graphrag", "automation"],
    default_args={
        "owner": "airflow",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }
)
def daily_update_dag():
    
    @task()
    def get_active_channels():
        """Получает список каналов из .env.scraper И базы данных."""
        from dotenv import dotenv_values
        ENV_DIR = "/opt/airflow"
        ENV_PATH = os.path.join(ENV_DIR, '.env.scraper')
        
        unique_channels = {} # {username: url}

        # 1. Читаем из .env.scraper
        if os.path.exists(ENV_PATH):
            config = dotenv_values(ENV_PATH)
            ch_list = config.get("CHANNELS_LIST", "").split(",")
            for c in ch_list:
                c = c.strip()
                if not c: continue
                # Улучшенная логика получения username (как в боте)
                username = c.lower().replace("https://", "").replace("http://", "")
                username = username.replace("t.me/s/", "").replace("t.me/", "").replace("@", "")
                username = username.split("/")[0].split("?")[0]
                
                unique_channels[username] = c
            print(f"--- Загружено {len(unique_channels)} уникальных каналов из .env.scraper ---")

        # 2. Читаем из БД
        with SessionLocal() as session:
            query = select(ProcessingChannel.username, ProcessingChannel.url).where(ProcessingChannel.status != 'ERROR')
            db_res = session.execute(query).all()
            db_count = 0
            for c in db_res:
                user_key = c.username.lower()
                if user_key not in unique_channels:
                    unique_channels[user_key] = c.url or f"@{c.username}"
                    db_count += 1
            print(f"--- Добавлено {db_count} новых каналов из базы данных ---")

        # Формируем итоговый список
        result = [{"username": u, "url": url} for u, url in unique_channels.items()]
        print(f"ИТОГО К ОБРАБОТКЕ: {len(result)} каналов.")
        return result

    @task()
    def update_channels_task(channels):
        """Последовательно обновляет каждый канал."""
        if not channels:
            print("No active channels found for update.")
            return

        print(f"Starting daily update for {len(channels)} channels.")
        
        # Настройки API
        API_ID = int(os.getenv("TG_API_ID"))
        API_HASH = os.getenv("TG_API_HASH")
        SESSION_PATH = os.path.join(PROJECT_ROOT, "anon.session")
        
        # Словарь для итоговой статистики
        stats = {}

        async def process_all():
            client = TelegramClient(SESSION_PATH, API_ID, API_HASH)
            await client.start()
            
            for channel in channels:
                username = channel['username']
                url = channel['url']
                print(f"--- Updating channel: @{username} ---")
                
                try:
                    # 1. Скрапинг с умным лимитом
                    update_db_status(username, "SCRAPING", 10)
                    with SessionLocal() as db_session:
                        # Проверяем, есть ли новости по этому каналу в базе
                        news_exists = db_session.query(News).filter(News.source == username.lower()).first()
                        if not news_exists:
                            current_limit = int(os.getenv("NEW_CHANNEL_LIMIT", 200))
                            print(f"Channel @{username} is NEW (0 news in DB). Limit: {current_limit}")
                        else:
                            current_limit = int(os.getenv("DAILY_UPDATE_LIMIT", 100))
                            print(f"Channel @{username} is ACTIVE. Limit: {current_limit}")
                        
                        new_count = await scrape_channel(client, db_session, url, limit=current_limit)
                        stats[username] = new_count
                    update_db_status(username, "SCRAPING", 40)
                    
                    # 2. Экстракция
                    update_db_status(username, "EXTRACTING", 50)
                    with SessionLocal() as session:
                        # Берем все необработанные новости для этого канала
                        query = select(News).where(News.source == username.lower(), News.is_processed == False)
                        batch = list(session.execute(query).scalars().all())
                        
                        if batch:
                            print(f"Extracting entities for {len(batch)} new messages in @{username}...")
                            await graph.ainvoke({"batch": batch, "channel_username": username})
                            update_db_status(username, "ACTIVE", 100)
                        else:
                            print(f"No new messages for @{username}.")
                            update_db_status(username, "ACTIVE", 100)
                            
                except Exception as e:
                    print(f"Error updating channel @{username}: {e}")
                    update_db_status(username, "ERROR", 0, error_message=str(e))
                
                await asyncio.sleep(5)
            
            await client.disconnect()

        asyncio.run(process_all())
        
        # --- ФИНАЛЬНЫЙ ОТЧЕТ ---
        summary_lines = []
        total_all_news = 0
        for u, count in stats.items():
            summary_lines.append(f"• {u}: +{count} новостей")
            total_all_news += count
        
        summary_text = "\n".join(summary_lines)
        report = (
            "🔄 *Ежедневное обновление завершено!*\n\n"
            "📊 *Статистика по каналам:*\n"
            f"{summary_text}\n\n"
            f"🚀 *Всего добавлено: {total_all_news} новостей*"
        )
        
        print("\n" + "="*30)
        print("📊 ИТОГИ ОБНОВЛЕНИЯ:")
        print(summary_text)
        print(f"ИТОГО: {total_all_news}")
        print("="*30 + "\n")
        
        # Отправка в бот
        if TG_BOT_TOKEN and ADMIN_CHAT_ID:
            try:
                bot_url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
                payload = {
                     "chat_id": ADMIN_CHAT_ID,
                     "text": report,
                     "parse_mode": "Markdown"
                }
                resp = requests.post(bot_url, json=payload, timeout=10)
                if resp.status_code == 200:
                    print("✅ Отчет успешно отправлен в Telegram.")
                else:
                    print(f"⚠️ Ошибка отправки в Telegram: {resp.status_code}")
            except Exception as e:
                print(f"⚠️ Не удалось отправить уведомление в бот: {e}")

    # Определение цепочки
    active_channels = get_active_channels()
    update_channels_task(active_channels)

daily_update_dag()
