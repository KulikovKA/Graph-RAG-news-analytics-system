import os
import sys
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from telethon import TelegramClient, events

# Настройка путей
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.common.models import Base, News

# Загрузка конфига
load_dotenv(os.path.join(PROJECT_ROOT, '.env.scraper'))

# Настройки БД
user = os.getenv('DB_USER', 'postgres')
password = os.getenv('DB_PASSWORD', 'mysecretpassword')
host = os.getenv('DB_HOST', 'postgres')
port = os.getenv('DB_PORT', '5432')
db_name = os.getenv('DB_NAME', 'graph_rag')
DB_URL = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

# Настройки Telegram API
API_ID = int(os.getenv("TG_API_ID"))
API_HASH = os.getenv("TG_API_HASH")
PHONE = os.getenv("TG_PHONE")
NEW_CHANNEL_LIMIT = int(os.getenv("NEW_CHANNEL_LIMIT", 200))
DAILY_UPDATE_LIMIT = int(os.getenv("DAILY_UPDATE_LIMIT", 20))
CHANNELS_LIST = [c.strip() for c in os.getenv("CHANNELS_LIST", "").split(",") if c.strip()]

# Путь для файла сессии
SESSION_PATH = os.path.join(PROJECT_ROOT, "anon.session")

async def scrape_channel(client, db_session, channel_link, limit=None):
    try:
        fetch_limit = limit if limit is not None else DAILY_UPDATE_LIMIT
        print(f"[{channel_link}] Начало сбора через API (лимит: {fetch_limit})...")
        # Убираем возможные префиксы для Telethon
        name = channel_link.lower()
        for prefix in ["https://t.me/s/", "http://t.me/s/", "t.me/s/", "https://t.me/", "http://t.me/", "t.me/", "@"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        
        entity = await client.get_entity(name)
        new_count = 0
        already_in_db = 0
        
        # Получаем последние сообщения с учётом заданного лимита
        async for message in client.iter_messages(entity, limit=fetch_limit):
            if not message.text:
                continue
            
            # Проверяем в базе (ID сообщений в API - это простые Integer)
            exists = db_session.query(News).filter_by(source=name, message_id=message.id).first()
            if not exists:
                new_entry = News(
                    message_id=message.id,
                    source=name,
                    text=message.text,
                    created_at=message.date.replace(tzinfo=None) if message.date else datetime.utcnow(),
                    is_processed=False
                )
                db_session.add(new_entry)
                new_count += 1
            else:
                already_in_db += 1
        
        db_session.commit()
        print(f"[{name}] Итог: Добавлено {new_count}, Уже в базе: {already_in_db}")
        return new_count
        
    except Exception as e:
        db_session.rollback()
        print(f"[{channel_link}] Ошибка API-парсинга: {e}")
        return 0

async def main():
    print(f"Запуск API-парсера (ID: {API_ID})...")
    client = TelegramClient(SESSION_PATH, API_ID, API_HASH)
    
    await client.start(phone=PHONE)
    
    if not await client.is_user_authorized():
        print("Ошибка: Не удалось авторизоваться.")
        return

    print("Авторизация успешна! Начинаю парсинг...")
    
    db_session = SessionLocal()
    try:
        for channel in CHANNELS_LIST:
            await scrape_channel(client, db_session, channel)
    finally:
        db_session.close()
        await client.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
