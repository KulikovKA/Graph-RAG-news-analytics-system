import os
import sys
import argparse
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

# Настройка путей
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, '.env.db'))

# 1. Postgres Setup
user = os.getenv('POSTGRES_USER', 'postgres')
password = os.getenv('POSTGRES_PASSWORD', 'mysecretpassword')
db_name = os.getenv('POSTGRES_DB', 'graph_rag')
host = os.getenv('POSTGRES_HOST', 'localhost').strip()
port = os.getenv('POSTGRES_PORT', '5432').strip()

DB_URL = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

from src.common.models import ProcessingChannel, News

def cleanup(username):
    print(f"--- Начинаю полную очистку для канала: @{username} ---")
    
    # --- Postgres Cleanup ---
    session = SessionLocal()
    try:
        # Удаляем новости
        news_deleted = session.query(News).filter(News.source == username).delete()
        # Удаляем запись о процессе
        channel_deleted = session.query(ProcessingChannel).filter_by(username=username).delete()
        session.commit()
        print(f"[Postgres] Удалено новостей: {news_deleted}, Записей мониторинга: {channel_deleted}")
    except Exception as e:
        session.rollback()
        print(f"[Postgres] Error: {e}")
    finally:
        session.close()

    # --- Neo4j Cleanup ---
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "mysecretpassword")
    
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            # Удаляем связи, созданные этим каналом
            result = session.run(
                "MATCH ()-[r]->() WHERE r.source = $username DELETE r RETURN count(r) as count",
                username=username
            )
            rel_count = result.single()["count"]
            print(f"[Neo4j] Удалено связей: {rel_count}")
        driver.close()
    except Exception as e:
        print(f"[Neo4j] Error: {e}")

    # --- Qdrant Cleanup ---
    try:
        q_host = os.getenv("QDRANT_HOST", "localhost")
        q_port = int(os.getenv("QDRANT_PORT", "6333"))
        client = QdrantClient(host=q_host, port=q_port)
        
        if client.collection_exists("news_segments"):
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            
            # Удаляем по фильтру payload.source
            delete_result = client.delete(
                collection_name="news_segments",
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=username),
                        )
                    ]
                ),
            )
            print(f"[Qdrant] Эмбеддинги для @{username} удалены (status: {delete_result.status})")
        else:
            print("[Qdrant] Коллекция news_segments не найдена.")
    except Exception as e:
        print(f"[Qdrant] Error: {e}")

    print("--- Очистка завершена. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full cleanup of channel data across all DBs")
    parser.add_argument("username", help="Telegram channel username (without @)")
    args = parser.parse_args()
    
    cleanup(args.username)
