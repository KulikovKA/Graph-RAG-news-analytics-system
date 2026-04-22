import os
import sys
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
from dotenv import load_dotenv
import mlflow
import time
import json
import re
from collections import Counter

# Патч: aiohttp 3.9.x убрал ClientConnectorDNSError, но google-genai SDK его требует
import aiohttp
if not hasattr(aiohttp, 'ClientConnectorDNSError'):
    aiohttp.ClientConnectorDNSError = aiohttp.ClientConnectorError

from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.processor.text_utils import normalize_text, ensure_string
from src.common.telemetry import telemetry
from src.config.config_retrieval import (
    TOP_K_MODE_1, 
    TOP_K_SEARCH_MODE_2, 
    TOP_K_PROMPT_MODE_2,
    TOP_K_SEARCH_MODE_3,
    TOP_K_PROMPT_MODE_3,
    SAFE_TPM_THRESHOLD,
    MAX_ENTITIES_ANALYTICAL,
    CHUNK_THRESHOLD
)
from src.config.config_models import (
    MODEL_MODE_1, 
    MODEL_MODE_2, 
    MODEL_MODE_3, 
    MODEL_HELPER,
    MODEL_WEB_SYNTHESIS
)

# Константы для Fallback
SIGNAL_GAP_DETECTED = "SIGNAL_GAP_DETECTED"
GAP_VECTOR_COUNT_THRESHOLD = 3
GAP_VECTOR_LENGTH_THRESHOLD = 1500

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, '.env.db'))

class GraphRAGSearcher:
    """Базовый поисковик (Mode 1)."""
    def __init__(self):
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_pwd = os.getenv("NEO4J_PASSWORD", "mysecretpassword")
        self.neo4j_driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pwd))
        
        qdrant_url = os.getenv("QDRANT_URL")
        if not qdrant_url:
            q_host = os.getenv("QDRANT_HOST", "localhost")
            q_port = os.getenv("QDRANT_PORT", "6333")
            qdrant_url = f"http://{q_host}:{q_port}"
        
        self.qdrant = AsyncQdrantClient(url=qdrant_url)
        self._embeddings = None
        
        try:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
            mlflow.set_experiment("GraphRAG_expirements")
        except Exception: pass
        
        self.llm_local = ChatGoogleGenerativeAI(
            model=MODEL_MODE_1, google_api_key=os.getenv("GOOGLE_API_KEY"), 
            temperature=0.1, streaming=True, max_retries=0, timeout=40
        )
        self.llm_web = ChatGoogleGenerativeAI(
            model=MODEL_WEB_SYNTHESIS, google_api_key=os.getenv("GOOGLE_API_KEY"), 
            temperature=0.1, streaming=True, max_retries=0, timeout=40
        )
        self.llm_helper = ChatGoogleGenerativeAI(
            model=MODEL_HELPER, google_api_key=os.getenv("GOOGLE_API_KEY"), 
            temperature=0.0, max_retries=0
        )
        tavily_key = os.getenv("TAVILY_API_KEY")
        self.tavily = None
        if tavily_key:
            from langchain_community.tools.tavily_search import TavilySearchResults
            self.tavily = TavilySearchResults(api_key=tavily_key, max_results=5)

    async def web_search(self, query: str, depth: str = "basic") -> str:
        """Поиск в глобальном интернете через Tavily (Fallback)."""
        if not self.tavily:
            return "⚠️ Tavily API Key не настроен. Веб-поиск недоступен."
        try:
            results = await self.tavily.ainvoke({"query": query})
            
            if not results:
                return "В глобальном интернете информации по данному запросу не обнаружено."
            
            # Если результат уже строка (инструмент отформатировал сам), возвращаем её
            if isinstance(results, str):
                return results

            # Если результат - список (стандарт для langchain_tavily)
            if isinstance(results, list):
                formatted_results = []
                for res in results:
                    if isinstance(res, dict):
                        url = res.get('url', 'URL неизвестен')
                        content = res.get('content', '')
                        formatted_results.append(f"🔗 {url}\n📝 {content}")
                    elif isinstance(res, str):
                        formatted_results.append(res)
                
                return "\n\n".join(formatted_results)
            
            return str(results)
        except Exception as e:
            return f"⚠️ Ошибка внешнего поиска (Tavily): {str(e)}"


    @property
    def embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base", model_kwargs={'device': 'cpu'})
        return self._embeddings

    def _estimate_tokens(self, text: str) -> int: return int(len(text) / 3.0)

    def _trim_context(self, texts: List[str], overhead_context: str = "") -> str:
        overhead_tokens = self._estimate_tokens(overhead_context)
        allowed_tokens = SAFE_TPM_THRESHOLD - overhead_tokens - 500
        current_len = 0; trimmed_texts = []
        for t in texts:
            if current_len + len(t) < allowed_tokens * 3:
                trimmed_texts.append(t); current_len += len(t)
            else: break
        if not trimmed_texts and texts: return f"- {texts[0][:allowed_tokens * 4]}"
        return "\n".join([f"- {t}" for t in trimmed_texts])

    def _get_active_sources(self) -> List[str]:
        try:
            scraper_env = {}
            scraper_path = os.path.join(PROJECT_ROOT, '.env.scraper')
            if os.path.exists(scraper_path):
                with open(scraper_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '=' in line and not line.startswith('#'):
                            k, v = line.strip().split('=', 1)
                            scraper_env[k] = v.strip("'").strip('"')
            all_channels = [c.strip() for c in scraper_env.get("CHANNELS_LIST", "").split(",") if c.strip()]
            disabled = [c.strip() for c in scraper_env.get("DISABLED_CHANNELS_LIST", "").split(",") if c.strip()]
            active = [c for c in all_channels if c not in disabled]
            normalized_active = []
            for c in active:
                clean = c.replace("https://t.me/", "").replace("t.me/", "").strip("@").strip("/")
                normalized_active.append(clean); normalized_active.append(c)
            return list(set(normalized_active))
        except Exception: return []

    def _get_llm_info(self, obj):
        """Разбирает цепочки LangChain и находит информацию о модели."""
        targets = []
        if hasattr(obj, 'steps'): targets = obj.steps
        elif hasattr(obj, 'middle'): targets = [obj.middle]
        elif hasattr(obj, 'bound'): targets = [obj.bound]
        else: targets = [obj]
        
        for t in targets:
            # Сначала проверяем явное соответствие нашим инстансам
            if hasattr(self, 'llm_local') and t == self.llm_local: return MODEL_MODE_1, "Mode 1"
            if hasattr(self, 'llm_mode_2') and t == self.llm_mode_2: return MODEL_MODE_2, "Mode 2"
            if hasattr(self, 'llm_mode_3') and t == self.llm_mode_3: return MODEL_MODE_3, "Mode 3"
            if hasattr(self, 'llm_web') and t == self.llm_web: return MODEL_WEB_SYNTHESIS, "Web Synthesis"
            if hasattr(self, 'llm_helper') and t == self.llm_helper: return MODEL_HELPER, "Helper"
            
            # Если нет прямого матча, пробуем вытащить имя из атрибутов
            if hasattr(t, 'model_name'): return t.model_name, "LLM"
            if hasattr(t, 'model'): return t.model, "LLM"
        
        return MODEL_MODE_1, "Mode 1" # Дефолт-заглушка

    async def _handle_llm_error(self, e: Exception, attempt: int, delays: list, on_status=None):
        err_str = str(e).lower()
        is_quota = any(x in err_str for x in ["429", "resource_exhausted", "quota"])
        is_network = any(x in err_str for x in ["connect", "disconnect", "timeout", "network", "unreachable", "ssl"])
        is_unavailable = "503" in err_str or "unavailable" in err_str
        if is_unavailable: return "unavailable", "⚠️ Сервера сейчас перегружены (503). Попробуйте позже."
        if (is_quota or is_network) and attempt < len(delays):
            wait_time = delays[attempt]
            if is_quota:
                retry_match = re.search(r"retry in ([\d.]+)s", err_str)
                if retry_match: wait_time = float(retry_match.group(1)) + 1
            if on_status: await on_status(f"🤖 ПРЕВЫШЕН ЛИМИТ/СБОЙ. Ждем {wait_time:.1f} сек...")
            await asyncio.sleep(wait_time); return "retry", ""
        return "fail", str(e)

    async def _invoke_with_retry(self, chain: Any, input_data: Any, on_status: Any = None) -> Any:
        delays = [5, 10, 20, 40]
        model_name, mode = (MODEL_HELPER, "Helper") if isinstance(chain, str) else self._get_llm_info(chain)
        
        for i in range(len(delays) + 1):
            try:
                if isinstance(chain, str): 
                    resp = await self.llm_helper.ainvoke(chain)
                else:
                    resp = await chain.ainvoke(input_data)
                
                telemetry.record_request(model_name, mode, "success")
                
                if hasattr(resp, 'response_metadata') and 'token_usage' in resp.response_metadata:
                    usage = resp.response_metadata['token_usage']
                    telemetry.record_tokens(model_name, mode, 
                                           prompt_tokens=usage.get('prompt_tokens', 0), 
                                           completion_tokens=usage.get('completion_tokens', 0))
                else:
                    telemetry.record_tokens(model_name, mode, 
                                           prompt_tokens=self._estimate_tokens(str(input_data)), 
                                           completion_tokens=self._estimate_tokens(ensure_string(resp.content if hasattr(resp, 'content') else resp)))
                return resp
            except Exception as e:
                telemetry.record_request(model_name, mode, "error")
                action, msg = await self._handle_llm_error(e, i, delays, on_status)
                if action == "retry": continue
                raise Exception(msg)

    async def _stream_with_retry(self, chain: Any, input_data: Dict[str, Any], on_status: Any = None) -> AsyncGenerator[str, None]:
        delays = [10, 20, 30, 60]
        model_name, mode = self._get_llm_info(chain)

        for i in range(len(delays) + 1):
            try:
                full_content = ""
                async for chunk in chain.astream(input_data):
                    full_content += chunk
                    yield chunk
                
                telemetry.record_request(model_name, mode, "success")
                telemetry.record_tokens(model_name, mode, 
                                       prompt_tokens=self._estimate_tokens(str(input_data)), 
                                       completion_tokens=self._estimate_tokens(full_content))
                return
            except Exception as e:
                telemetry.record_request(model_name, mode, "error")
                action, msg = await self._handle_llm_error(e, i, delays, on_status)
                if action == "retry": continue
                raise Exception(msg)

    async def search_qdrant(self, query: str, top_k: int = TOP_K_MODE_1, use_filter: bool = True) -> str:
        vector = await self.embeddings.aembed_query(query)
        active_sources = self._get_active_sources() if use_filter else []
        query_filter = None
        if active_sources:
            query_filter = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchAny(any=active_sources))])
        response = await self.qdrant.query_points(collection_name="news_segments", query=vector, query_filter=query_filter, limit=top_k, score_threshold=CHUNK_THRESHOLD)
        if not response.points and use_filter and active_sources: return await self.search_qdrant(query, top_k, use_filter=False)
        if not response.points: return "Сырые тексты новостей не найдены."
        return "\n".join([f"- {hit.payload['text']} (Источник: {hit.payload['source']})" for hit in response.points])

    async def search_neo4j(self, entities: List[str], use_filter: bool = True) -> str:
        active_sources = self._get_active_sources() if use_filter else []
        cypher_query = """
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE (n.id IN $entities OR n.name IN $entities OR ANY(e IN $entities WHERE n.name CONTAINS e))
              AND (size($active_sources) = 0 OR r.sources IS NULL OR ANY(src IN r.sources WHERE src IN $active_sources))
        RETURN n.name AS source, type(r) AS relation, m.name AS target, r.value AS value, r.sentiment AS sentiment
        LIMIT 40
        """
        async with self.neo4j_driver.session() as session:
            result = await session.run(cypher_query, entities=entities, active_sources=active_sources)
            records = await result.data()
        if not records: return "Прямых связей в графе не обнаружено."
        facts = []
        for rec in records:
            facts.append(f"{rec['source']} --{rec['relation']}-> {rec['target']}")
        return "\n".join(facts)

    async def extract_entities_from_query(self, query: str) -> List[str]:
        prompt = f"Извлеки ключевые сущности через запятую: {query}"
        try:
            resp = await self._invoke_with_retry(prompt, None)
            content = ensure_string(resp.content if hasattr(resp, 'content') else resp)
            return [normalize_text(ent) for ent in content.split(",") if ent.strip() and len(ent.strip()) > 2]
        except Exception: return []

    async def ask(self, query: str, on_status=None, web_context: str = None) -> AsyncGenerator[str, None]:
        start_time = time.perf_counter(); full_answer = ""
        with mlflow.start_run(run_name=f"Mode 1: {query[:30]}"):
            mlflow.log_param("query", query)
            mlflow.log_param("mode", "Mode 1")
            try:
                if not web_context:
                    if on_status: await on_status("🔍 Анализ вопроса...")
                    entities = await self.extract_entities_from_query(query)
                    graph_context, vector_context_raw = await asyncio.gather(self.search_qdrant(query), self.search_neo4j(entities))
                    q_points = await self.qdrant.query_points(collection_name="news_segments", query=await self.embeddings.aembed_query(query), limit=TOP_K_MODE_1)
                    texts = [hit.payload['text'] for hit in q_points.points]
                    vector_context = self._trim_context(texts, graph_context)
                    if "не обнаружено" in graph_context.lower() and vector_context.count("- ") < GAP_VECTOR_COUNT_THRESHOLD:
                        yield SIGNAL_GAP_DETECTED; return
                else: vector_context = graph_context = "Локальные данные не использованы."
                
                mlflow.log_text(graph_context, "graph_context.txt")
                mlflow.log_text(vector_context, "vector_context.txt")
                
                prompt = ChatPromptTemplate.from_template("Ты ассистент GraphRAG.\nГРАФ: {graph_context}\nТЕКСТЫ: {vector_context}\nWEB: {web_context}\nВопрос: {query}")
                chain = prompt | (self.llm_web if web_context else self.llm_local) | StrOutputParser()
                async for chunk in self._stream_with_retry(chain, {"graph_context": graph_context, "vector_context": vector_context, "web_context": web_context or "Нет данных", "query": query}, on_status):
                    full_answer += chunk; yield chunk
                
                mlflow.log_text(full_answer, "answer.txt")
            except Exception as e: 
                mlflow.log_param("error", str(e))
                yield f"⚠️ Ошибка: {str(e)}"

class AdvancedAnalyticalRouter(GraphRAGSearcher):
    def __init__(self):
        super().__init__()
        self.llm_mode_2 = ChatGoogleGenerativeAI(model=MODEL_MODE_2, google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.1, streaming=True, max_retries=0, timeout=60)
        self.llm_mode_3 = ChatGoogleGenerativeAI(model=MODEL_MODE_3, google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.1, streaming=True, max_retries=0, timeout=60)

    async def get_ecosystem_map(self, center_id: str) -> str:
        active_sources = self._get_active_sources()
        query = "MATCH (n:Entity {id: $center_id})-[r*..2]-(m:Entity) LIMIT 25"
        async with self.neo4j_driver.session() as session:
            result = await session.run(query, center_id=center_id); records = await result.data()
        return "ЭКОСИСТЕМА:\n" + "\n".join([f"{r['neighbor']}" for r in records]) if records else ""

    async def ask(self, query: str, on_status=None, web_context: str = None) -> AsyncGenerator[str, None]:
        start_time = time.perf_counter(); full_answer = ""
        with mlflow.start_run(run_name=f"Mode 2: {query[:30]}"):
            mlflow.log_param("query", query)
            mlflow.log_param("mode", "Mode 2")
            try:
                if not web_context:
                    if on_status: await on_status("📡 Глубокий поиск...")
                    vector = await self.embeddings.aembed_query(query)
                    q_res = await self.qdrant.query_points(collection_name="news_segments", query=vector, limit=TOP_K_SEARCH_MODE_2, score_threshold=0.18)
                    texts = [hit.payload['text'] for hit in q_res.points]
                    all_texts = "\n".join(texts)
                    resp = await self._invoke_with_retry(f"Извлеки компании:\n{all_texts[:8000]}", None)
                    content = ensure_string(resp.content if hasattr(resp, 'content') else resp)
                    top_entities = [normalize_text(e) for e in content.split(",") if e.strip()][:MAX_ENTITIES_ANALYTICAL]
                    graph_context = await self.search_neo4j(top_entities)
                    vector_context = self._trim_context(texts, graph_context)
                    if "не обнаружено" in graph_context.lower() and len(texts) < (GAP_VECTOR_COUNT_THRESHOLD + 1):
                        yield SIGNAL_GAP_DETECTED; return
                else: graph_context = vector_context = "Локальные данные не использованы."
                
                mlflow.log_text(graph_context, "graph_context.txt")
                mlflow.log_text(vector_context, "vector_context.txt")
                
                prompt = ChatPromptTemplate.from_template("Ты аналитик.\nГРАФ: {graph_context}\nТЕКСТЫ: {vector_context}\nWEB: {web_context}\nВопрос: {query}")
                chain = prompt | self.llm_mode_2 | StrOutputParser()
                async for chunk in self._stream_with_retry(chain, {"graph_context": graph_context, "vector_context": vector_context, "web_context": web_context or "Нет данных", "query": query}, on_status):
                    full_answer += chunk; yield chunk
                
                mlflow.log_text(full_answer, "answer.txt")
            except Exception as e: 
                mlflow.log_param("error", str(e))
                yield f"⚠️ Ошибка: {str(e)}"

    async def generate_digest(self, topic: str, date_input: str, on_status=None, web_context: str = None) -> AsyncGenerator[str, None]:
        start_time = time.perf_counter(); full_answer = ""
        with mlflow.start_run(run_name=f"Mode 3: {topic[:30]}"):
            mlflow.log_param("topic", topic)
            mlflow.log_param("date_range", date_input)
            mlflow.log_param("mode", "Mode 3")
            try:
                if not web_context:
                    if on_status: await on_status("📅 Валидация дат...")
                    start_dt, end_dt = self._parse_dates(date_input)
                    q_start_ts = start_dt.timestamp()
                    q_end_ts = end_dt.replace(hour=23, minute=59, second=59).timestamp()
                    vector = await self.embeddings.aembed_query(topic)
                    must = [models.FieldCondition(key="timestamp", range=models.Range(gte=q_start_ts, lte=q_end_ts))]
                    q_res = await self.qdrant.query_points(collection_name="news_segments", query=vector, query_filter=models.Filter(must=must), limit=TOP_K_SEARCH_MODE_3, score_threshold=0.35)
                    if not q_res.points:
                        yield f"🕵️ За период {date_input} новостей по теме '{topic}' не найдено."; return
                    texts = []
                    for hit in q_res.points:
                        p_text = hit.payload.get('text', '')
                        p_ts = hit.payload.get('timestamp')
                        dt_str = datetime.fromtimestamp(p_ts).strftime("%d.%m.%Y") if isinstance(p_ts, (int, float)) else str(p_ts)
                        texts.append(f"[{dt_str}] {p_text}")
                    texts = texts[:TOP_K_PROMPT_MODE_3]
                    if on_status: await on_status("🧬 Анализ связей...")
                    resp = await self._invoke_with_retry(f"Извлеки компании:\n{chr(10).join(texts)[:4000]}", None)
                    content = ensure_string(resp.content if hasattr(resp, 'content') else resp)
                    entities = [normalize_text(e) for e in content.split(",") if e.strip()]
                    graph_context = await self.search_neo4j(list(set(entities))[:10])
                    vector_context = self._trim_context(texts, graph_context)
                else: graph_context = vector_context = "Локальные данные не использованы."
                
                mlflow.log_text(graph_context, "graph_context.txt")
                mlflow.log_text(vector_context, "vector_context.txt")
                
                prompt = ChatPromptTemplate.from_template("""
ТЫ — БЕЗЖАЛОСТНЫЙ НОВОСТНОЙ КОРРЕКТОР. 
Твоя задача: составить ЧИСТЫЙ дайджест строго по заданной теме.

ВВОДНЫЕ ДАННЫЕ (Запрос): {query}

СЫРЫЕ ТЕКСТЫ (ВНИМАНИЕ: может содержать ШУМ!):
{vector_context}

ГРАФОВЫЕ СВЯЗИ:
{graph_context}

АЛГОРИТМ ТВОЕЙ РАБОТЫ (НАРУШЕНИЕ ПРАВИЛ НЕДОПУСТИМО):
1. ПРАВИЛО ОТРАСЛЕВОЙ ИЗОЛЯЦИИ: Если запрос касается авторынка ("автомобили", "шины"), КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО включать новости про:
   - Косметику, парфюмерию, одежду, продукты питания (ДАЖЕ если там есть слова "пошлина", "цены" или "ЕАЭС").
   - Железнодорожный транспорт ("вагоны", "подвижной состав"), авиацию.
   - Мелкие ДТП, пробки и любые криминальные угоны одиночных машин.
   Это ШУМ. Удаляй эти новости полностью.

2. ПРОВЕРКА СМЫСЛА: Удаляй обрывки фраз без начала или конца. Если предложение не несет законченного факта — выбрасывай.

3. ПРАВИЛО РЕЛЕВАНТНОСТИ: Включай только те факты, которые ПРЯМО отвечают на вопрос "{query}". 

ОФОРМЛЕНИЕ (СТРОГО):
# Хроника событий: [Тема]
**Краткое резюме:** [1-2 предложения сути найденного]

### Лента событий:
* [Дата]: [ОДНО полное и логичное предложение с фактом].
(Группируй факты за одну дату в один пункт).

Вопрос: {query}
""")
                chain = prompt | self.llm_mode_3 | StrOutputParser()
                async for chunk in self._stream_with_retry(chain, {"query": topic, "graph_context": graph_context, "vector_context": vector_context}, on_status):
                    full_answer += chunk; yield chunk
                
                mlflow.log_text(full_answer, "answer.txt")
            except Exception as e: 
                mlflow.log_param("error", str(e))
                yield f"⚠️ Ошибка: {str(e)}"

    async def _get_available_date_range(self) -> tuple:
        try:
            for key in ["timestamp", "date"]:
                try:
                    res_min, _ = await self.qdrant.scroll(collection_name="news_segments", limit=1, with_payload=True, order_by=models.OrderBy(key=key, direction=models.Direction.ASC))
                    res_max, _ = await self.qdrant.scroll(collection_name="news_segments", limit=1, with_payload=True, order_by=models.OrderBy(key=key, direction=models.Direction.DESC))
                    if res_min and res_max:
                        min_dt = datetime.fromtimestamp(res_min[0].payload.get(key)).date() if isinstance(res_min[0].payload.get(key), (int, float)) else datetime.fromisoformat(res_min[0].payload.get(key).replace(" ", "T")).date()
                        max_dt = datetime.fromtimestamp(res_max[0].payload.get(key)).date() if isinstance(res_max[0].payload.get(key), (int, float)) else datetime.fromisoformat(res_max[0].payload.get(key).replace(" ", "T")).date()
                        return min_dt.strftime("%d.%m.%Y"), max_dt.strftime("%d.%m.%Y")
                except: continue
            return None, None
        except Exception: return None, None

    def _parse_dates(self, date_input: str) -> tuple:
        date_input = date_input.strip()
        try:
            if "-" in date_input:
                p = [x.strip() for x in date_input.split("-")]
                return datetime.strptime(p[0], "%d.%m.%Y"), datetime.strptime(p[1], "%d.%m.%Y")
            dt = datetime.strptime(date_input, "%d.%m.%Y")
            return dt, dt
        except Exception: raise ValueError("Формат: ДД.ММ.ГГГГ")
