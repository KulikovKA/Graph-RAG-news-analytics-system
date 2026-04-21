import os
from sqlalchemy import create_engine, text
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv('.env.db')

def wipe_postgres():
    print("--- Cleaning Postgres ---")
    user = os.getenv('POSTGRES_USER')
    password = os.getenv('POSTGRES_PASSWORD')
    host = os.getenv('POSTGRES_HOST', 'localhost')
    db = os.getenv('POSTGRES_DB')
    url = f"postgresql://{user}:{password}@{host}:5432/{db}"
    engine = create_engine(url)
    with engine.connect() as conn:
        # УДАЛЯЕМ все новости
        res1 = conn.execute(text("DELETE FROM news"))
        # УДАЛЯЕМ все каналы мониторинга
        res2 = conn.execute(text("DELETE FROM processing_channels"))
        conn.commit()
    print(f"Done. Deleted news: {res1.rowcount}, Deleted channels: {res2.rowcount}")

def wipe_neo4j():
    print("--- Cleaning Neo4j ---")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pw = os.getenv("NEO4J_PASSWORD", "mysecretpassword")
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()
    print("Done. Graph is empty.")

def wipe_qdrant():
    print("--- Cleaning Qdrant ---")
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    client = QdrantClient(host=host, port=port)
    if client.collection_exists("news_segments"):
        client.delete_collection("news_segments")
    print("Done. Qdrant collection deleted.")

if __name__ == "__main__":
    confirm = input("This will WIPE Neo4j, Qdrant and reset Postgres. Are you sure? (y/n): ")
    if confirm.lower() == 'y':
        wipe_postgres()
        wipe_neo4j()
        wipe_qdrant()
        print("\n✨ GLOBAL WIPE COMPLETED. You can now run the DAG.")
    else:
        print("Aborted.")
