import os
import sys
import asyncio
import json
import hashlib
import uuid
import time
import re

# Патч: aiohttp 3.9.x убрал ClientConnectorDNSError, но google-genai SDK его требует
import aiohttp
if not hasattr(aiohttp, 'ClientConnectorDNSError'):
    aiohttp.ClientConnectorDNSError = aiohttp.ClientConnectorError

from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine, update, and_
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
from pymystem3 import Mystem

# Тяжёлые ML-импорты подключаются лениво
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Настройка путей
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# Загрузка конфигурации
load_dotenv(os.path.join(PROJECT_ROOT, '.env.db'))
from src.common.models import ProcessingChannel, News
from src.config.config_models import MODEL_EXTRACTION
from src.config.config_retrieval import CHUNK_SIZE, CHUNK_OVERLAP

# Константы
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
MODEL_NAME = MODEL_EXTRACTION

# Настройки БД
user = os.getenv('POSTGRES_USER', 'postgres')
password = os.getenv('POSTGRES_PASSWORD', 'mysecretpassword')
host = os.getenv('POSTGRES_HOST', 'postgres')
port = os.getenv('POSTGRES_PORT', '5432')
db_name = os.getenv('POSTGRES_DB', 'graph_rag')
db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
engine = create_engine(db_url)
SessionLocal = sessionmaker(bind=engine)

from src.processor.text_utils import normalize_text, ensure_string

def generate_deterministic_id(source: str, message_id: int, chunk_idx: int = None) -> str:
    """Генерация стабильного UUID для новости или её чанка."""
    base_str = f"{source}:{message_id}"
    if chunk_idx is not None:
        base_str += f":chunk:{chunk_idx}"
    hash_hex = hashlib.sha256(base_str.encode()).hexdigest()
    return str(uuid.UUID(hash_hex[:32]))

# --- ПРОМПТЫ ---

EXTRACTION_PROMPT = """Ты — экспертный аналитик данных в сфере экономики и технологий.
Твоя цель: построить максимально детализированный граф знаний на основе СПИСКА новостей.
Анализируй тексты вместе, чтобы найти связи как внутри одной новости, так и МЕЖДУ ними.

1. СУЩНОСТИ (Nodes):
- Извлекай конкретику: ORGANIZATION, PERSON, LOCATION, EVENT, ASSET.
- Извлекай абстракции: INDUSTRY, PRODUCT, CONCEPT.
- "canonical_id": всегда в единственном числе, именительном падеже (например: Инфляция).

2. СВЯЗИ (Relationships):
- Ищи причинно-следственные связи.
- "news_indices": список индексов новостей из входного списка (например [0, 2]), к которым относится эта связь.

Формат вывода: только JSON.
{{
  "entities": [
    {{"name": "Имя", "label": "Тип", "canonical_id": "норма", "news_indices": [0]}}
  ],
  "relationships": [
    {{"source": "id1", "target": "id2", "type": "TYPE", "news_indices": [0, 1], "sentiment": 0.5, "value": 0.0}}
  ]
}}

СПИСОК НОВОСТЕЙ:
{text_list}
"""

SYNTHESIS_PROMPT = """Ты — главный аналитик. Твоя задача — дополнить граф знаний КРОСС-НОВОСТНЫМИ связями.
Тебе дан список из 12 новостей и сущности, которые в них упоминаются.

ЗАДАЧА:
1. Найди СЛОЖНЫЕ связи: например, новость №2 является следствием совокупности фактов из новостей №5 и №8.
2. Найди КРОСС-РЫНОЧНЫЕ влияния: как событие в одной новости влияет на актив/сектор из другой.
3. Не дублируй простые связи, ищи только глубокую аналитику между разными новостями из списка.

Формат вывода: только JSON.
{{
  "relationships": [
    {{"source": "id1", "target": "id2", "type": "TYPE", "news_indices": [0, 5], "sentiment": 0.5, "value": 0.0}}
  ]
}}

СПИСОК НОВОСТЕЙ:
{text_list}

УЖЕ НАЙДЕННЫЕ СУЩНОСТИ (используй их canonical_id для source/target):
{entities_list}
"""

# --- МОДЕЛИ ДАННЫХ ---

class Entity(BaseModel):
    name: str
    label: str
    canonical_id: str
    news_indices: List[int] = []

class Relationship(BaseModel):
    source: str
    target: str
    type: str
    news_indices: List[int] = []
    value: float = 0.0
    date: Optional[str] = None
    sentiment: float = 0.0

class ExtractionResult(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

class SynthesisResult(BaseModel):
    relationships: List[Relationship]

parser = JsonOutputParser(pydantic_object=ExtractionResult)
synthesis_parser = JsonOutputParser(pydantic_object=SynthesisResult)

# --- ЛЕНИВАЯ ЗАГРУЗКА ---

_llm = None
_embeddings = None
_neo4j_driver = None
_qdrant_client = None

def get_llm():
    global _llm
    if _llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0, timeout=300)
    return _llm

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    return _embeddings

def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        pw = os.getenv("NEO4J_PASSWORD", "mysecretpassword")
        _neo4j_driver = GraphDatabase.driver(uri, auth=(user, pw))
    return _neo4j_driver

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        _qdrant_client = QdrantClient(host=host, port=port)
    return _qdrant_client

def sanitize_rel_type(rel_type: str) -> str:
    clean = "".join(c for c in rel_type if c.isalnum() or c == '_').upper()
    return clean if clean else "RELATED_TO"

# --- ЛОГИКА ГРАФА ---

class GraphState(TypedDict):
    batch: List[Any]
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    channel_username: Optional[str]

async def extraction_node(state: GraphState):
    """Двухстадийная экстракция батчами по 12 новостей."""
    items = state["batch"]
    main_batch_size = 12
    main_batches = [items[i:i + main_batch_size] for i in range(0, len(items), main_batch_size)]
    
    username = state.get("channel_username")
    batch_pbar = tqdm(total=len(items), desc="  Batch Processing", leave=False, colour="green")
    processed_count = 0
    
    llm = get_llm()
    extract_prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT + "\n\n{format_instructions}")
    synth_prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT + "\n\n{format_instructions}")
    
    extract_chain = extract_prompt | llm
    synth_chain = synth_prompt | llm
    
    def update_db_progress(count):
        if not username: return
        try:
            progress_val = 60 + int((count / len(items)) * 35)
            with SessionLocal() as session:
                session.execute(update(ProcessingChannel).where(ProcessingChannel.username == username).values(progress=progress_val))
                session.commit()
        except Exception as e:
            print(f"Error updating progress: {e}")

    for m_batch in main_batches:
        # --- СЛОЙ 1: Параллельная экстракция (3 группы по 4) ---
        group_size = 4
        groups = [m_batch[i:i+group_size] for i in range(0, len(m_batch), group_size)]
        
        async def invoke_with_retry(chain, input_data):
            """Умный повтор для LLM вызовов при экстракции (сеть + квоты)."""
            delays = [5, 10, 20, 40, 60]
            for i, delay in enumerate(delays + [None]):
                try:
                    resp = await chain.ainvoke(input_data)
                    return resp
                except Exception as e:
                    err_str = str(e).lower()
                    is_quota = any(x in err_str for x in ["429", "resource_exhausted", "quota"])
                    is_network = any(x in err_str for x in ["connect", "disconnect", "timeout", "network", "unreachable", "ssl"])
                    
                    if (is_quota or is_network) and delay:
                        msg_type = "ПРЕВЫШЕН ЛИМИТ (429)" if is_quota else "ОШИБКА СВЯЗИ"
                        print(f"⚠️ [EXTRACTION_RETRY] {msg_type}. Ждем {delay} сек... (попытка {i+1})")
                        await asyncio.sleep(delay)
                        continue
                    raise e

        async def process_group(group_items):
            txt = "\n".join([f"Новость #{i}: {n.text}" for i, n in enumerate(group_items)])
            try:
                resp = await invoke_with_retry(extract_chain, {"text_list": txt, "format_instructions": parser.get_format_instructions()})
                res_dict = parser.parse(ensure_string(resp.content))
                return ExtractionResult(**res_dict)
            except Exception as e:
                print(f"  Group extraction failed after retries: {e}")
                return ExtractionResult(entities=[], relationships=[])

        # Запускаем параллельно
        layer1_tasks = [process_group(g) for g in groups]
        layer1_results = await asyncio.gather(*layer1_tasks)
        
        # Сбор промежутотных данных
        all_entities = []
        all_rels = []
        for r in layer1_results:
            all_entities.extend(r.entities)
            all_rels.extend(r.relationships)
            
        # Нормализация сущностей первого слоя
        for ent in all_entities: ent.canonical_id = normalize_text(ent.canonical_id)
        for rel in all_rels:
            rel.source = normalize_text(rel.source)
            rel.target = normalize_text(rel.target)

        # --- СЛОЙ 2: Глобальный синтез для батча из 12 ---
        full_txt = "\n".join([f"Новость #{i}: {n.text}" for i, n in enumerate(m_batch)])
        ent_list_str = ", ".join(set([e.canonical_id for e in all_entities]))
        
        synth_res = SynthesisResult(relationships=[])
        try:
            synth_resp = await invoke_with_retry(synth_chain, {
                "text_list": full_txt, 
                "entities_list": ent_list_str,
                "format_instructions": synthesis_parser.get_format_instructions()
            })
            synth_dict = synthesis_parser.parse(ensure_string(synth_resp.content))
            synth_res = SynthesisResult(**synth_dict)
            # Нормализация синтезированных связей
            for rel in synth_res.relationships:
                rel.source = normalize_text(rel.source)
                rel.target = normalize_text(rel.target)
        except Exception as e:
            print(f"  Synthesis layer failed: {e}")

        # --- ЗАПИСЬ ---
        driver = get_neo4j_driver()
        with driver.session() as session:
            # 1. Сначала сохраняем все узлы новостей батча
            for news_item in m_batch:
                det_id = generate_deterministic_id(news_item.source, news_item.message_id)
                session.run(
                    "MERGE (news:News {id: $news_id}) SET news.source = $source, news.date = $date",
                    news_id=det_id, source=news_item.source, date=str(news_item.created_at)
                )

            # 2. Запись сущностей и связей MENTIONED_IN
            for ent in all_entities:
                # Определяем, в каких новостях упомянута эта сущность
                # Для каждой новости из news_indices создаем связь MENTIONED_IN
                session.run("MERGE (n:Entity {id: $id}) SET n.name = $name, n.label = $label", id=ent.canonical_id, name=ent.name, label=ent.label)
                
                for idx in ent.news_indices:
                    if idx < len(m_batch):
                        news_item = m_batch[idx]
                        news_id = generate_deterministic_id(news_item.source, news_item.message_id)
                        session.run(
                            "MATCH (n:Entity {id: $ent_id}), (news:News {id: $news_id}) "
                            "MERGE (n)-[:MENTIONED_IN]->(news)",
                            ent_id=ent.canonical_id, news_id=news_id
                        )
            
            # 3. Запись связей (Слой 1 + Слой 2) с накоплением источников
            combined_rels = all_rels + synth_res.relationships
            for rel in combined_rels:
                rel_type = sanitize_rel_type(rel.type)
                # Берем первую новость из индексов как основной источник для метаданных
                idx = rel.news_indices[0] if rel.news_indices else 0
                news_item = m_batch[idx] if idx < len(m_batch) else m_batch[0]
                det_id = generate_deterministic_id(news_item.source, news_item.message_id)
                
                # Умный MERGE: накапливаем sources и det_ids в массивы
                session.run(
                    f"MATCH (a:Entity {{id: $aid}}) MATCH (b:Entity {{id: $bid}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    "ON CREATE SET r.value = $val, r.sentiment = $sent, r.sources = [$source], r.det_ids = [$det_id] "
                    "ON MATCH SET r.sources = CASE WHEN NOT $source IN r.sources THEN r.sources + $source ELSE r.sources END, "
                    "             r.det_ids = CASE WHEN NOT $det_id IN r.det_ids THEN r.det_ids + $det_id ELSE r.det_ids END", 
                    aid=rel.source, bid=rel.target, val=rel.value, sent=rel.sentiment, det_id=det_id, source=news_item.source
                )

        # Запись векторов и отметка в БД
        q_client = get_qdrant_client()
        # Проверка коллекции
        if not q_client.collection_exists("news_segments"):
            from qdrant_client.http.models import VectorParams, Distance
            q_client.create_collection(collection_name="news_segments", vectors_config=VectorParams(size=768, distance=Distance.COSINE))
            
        emb_model = get_embeddings()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
        )

        for news_item in m_batch:
            # Разбиваем текст новости на чанки
            chunks = text_splitter.split_text(news_item.text)
            news_id_base = generate_deterministic_id(news_item.source, news_item.message_id) # ID для Neo4j обратной совместимости
            
            points = []
            for i, chunk_text in enumerate(chunks):
                chunk_id = generate_deterministic_id(news_item.source, news_item.message_id, chunk_idx=i)
                vector = await emb_model.aembed_query(chunk_text)
                
                from qdrant_client.http.models import PointStruct
                points.append(PointStruct(
                    id=chunk_id, 
                    vector=vector, 
                    payload={
                        "text": chunk_text, 
                        "source": news_item.source, 
                        "message_id": news_item.message_id, 
                        "timestamp": news_item.created_at.timestamp(),
                        "chunk_idx": i,
                        "full_news_id": news_id_base
                    }
                ))
            
            if points:
                q_client.upsert(collection_name="news_segments", points=points)
            
            with SessionLocal() as db_session:
                db_session.execute(update(News).where(and_(News.source == news_item.source, News.message_id == news_item.message_id)).values(is_processed=True))
                db_session.commit()
            
            processed_count += 1
            batch_pbar.update(1)
            if processed_count % 3 == 0 or processed_count == len(items): update_db_progress(processed_count)

        # ПАУЗА МЕЖДУ БАТЧАМИ (Критически важно для выживания 15k TPM лимита)
        if processed_count < len(items):
            print(f"⏲️ Ожидание 40 сек перед следующим батчем для разгрузки TPM...")
            await asyncio.sleep(40)

    batch_pbar.close()
    return {"summary": {"processed": processed_count}}

workflow = StateGraph(GraphState)
workflow.add_node("extract", extraction_node)
workflow.set_entry_point("extract")
workflow.add_edge("extract", END)
graph = workflow.compile()
