import os
import re
import html
import sys
import asyncio
import logging
import time
import aiohttp
from aiogram import Bot, Dispatcher, types, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiohttp_socks import ProxyConnector
from dotenv import load_dotenv, set_key
import requests
from requests.auth import HTTPBasicAuth
from sqlalchemy import create_engine, select, update, and_
from sqlalchemy.orm import sessionmaker

# Настройка путей
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.bot.keyboards import (
    get_main_menu_keyboard, 
    get_modes_inline_keyboard, 
    get_sources_inline_keyboard,
    get_delete_sources_keyboard,
    get_web_fallback_keyboard,
    get_start_analysis_keyboard
)
from src.processor.query_service import (
    GraphRAGSearcher, 
    AdvancedAnalyticalRouter,
    SIGNAL_GAP_DETECTED
)

from src.config.config_models import MODEL_EXTRACTION
from src.common.models import Base, News, ProcessingChannel
from src.common.telemetry import telemetry

# Загрузка конфигурации
load_dotenv(os.path.join(PROJECT_ROOT, '.env.db'))
BOT_TOKEN = os.getenv("TG_BOT_TOKEN")

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройки БД (используем те же переменные, что и в миграции)
user = os.getenv('POSTGRES_USER', 'postgres')
password = os.getenv('POSTGRES_PASSWORD', 'mysecretpassword')
db_name = os.getenv('POSTGRES_DB', 'graph_rag')
host = os.getenv('POSTGRES_HOST', 'localhost').strip()
port = os.getenv('POSTGRES_PORT', '5432').strip()

DB_URL = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

# Настройки Airflow API
AIRFLOW_HOST = os.getenv("AIRFLOW_HOST", "localhost") # Переключается между Docker-именем и localhost
AIRFLOW_API_URL = f"http://{AIRFLOW_HOST}:8080/api/v1"
AIRFLOW_AUTH = HTTPBasicAuth("admin", "admin")

# Инициализация бота (будет проинициализирован в main)
bot = None
dp = Dispatcher()

# Глобальный трекер открытых дашбордов {chat_id: message_id}
active_sources_dashboards = {}

# --- Состояния FSM ---
class SourceManagement(StatesGroup):
    waiting_for_source_url = State()

class AnalysisState(StatesGroup):
    waiting_for_question = State()
    current_mode = State()

class DigestState(StatesGroup):
    waiting_for_topic = State()
    waiting_for_date = State()

# --- Вспомогательные функции ---
ENV_SCRAPER_PATH = os.path.join(PROJECT_ROOT, '.env.scraper')

def get_progress_bar(percent: int, length: int = 10) -> str:
    """Генерирует текстовый прогресс-бар."""
    if percent < 0: percent = 0
    if percent > 100: percent = 100
    filled = int(length * percent // 100)
    bar = "▓" * filled + "░" * (length - filled)
    return f"[{bar}] {percent}%"

def canonical_source_name(src: str) -> str:
    """Приводит имя канала к единому каноническому виду и очищает от лишних символов."""
    if not src: return ""
    res = src.strip().lower()
    # Убираем протоколы и домен
    res = res.replace("https://", "").replace("http://", "")
    res = res.replace("t.me/", "").replace("@", "")
    
    # Убираем все после слеша (подпапки)
    res = res.split("/")[0]
    
    # Убираем все после знака вопроса (параметры запроса)
    res = res.split("?")[0]
    
    # Регулярка для валидации имени пользователя Telegram (5-32 символа, латиница, цифры, _)
    if not re.match(r'^[a-z0-9_]{5,32}$', res):
        return None
        
    return res

async def safe_tg_call(coro, retries=3, delay=1):
    """Обертка для безопасных вызовов Telegram API с повторами при сетевых ошибках."""
    for i in range(retries):
        try:
            return await coro
        except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError, aiohttp.ClientOSError) as e:
            if i == retries - 1:
                raise e
            logger.warning(f"⚠️ Сетевая ошибка Telegram ({type(e).__name__}). Повтор {i+1}/{retries} через {delay}с...")
            await asyncio.sleep(delay)
        except Exception as e:
            err_str = str(e).lower()
            if "message is not modified" in err_str:
                return
            # Если ошибка в разметке (Markdown) - мы не можем её исправить повтором, 
            # передаем исключение выше для обработки (отправка без разметки)
            if "can't parse entities" in err_str or "can't find end" in err_str:
                # Отправляем сигнал, что разметка сломана
                return "ENTITY_ERROR"
            raise e

def sanitize_for_tg(text: str) -> str:
    """Экранирует HTML и переводит базовый Markdown ИИ в HTML-теги."""
    if not text: return ""
    # 1. Экранируем спецсимволы HTML
    safe_text = html.escape(text)
    # 2. **bold** -> <b>bold</b>
    safe_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', safe_text)
    # 3. *italic* -> <i>italic</i>
    safe_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', safe_text)
    # 4. `code` -> <code>code</code>
    safe_text = re.sub(r'`(.*?)`', r'<code>\1</code>', safe_text)
    return safe_text

def get_active_channels():
    """Читает список активных каналов из .env.scraper И базы данных (ACTIVE)."""
    from dotenv import dotenv_values
    config = dotenv_values(ENV_SCRAPER_PATH)
    channels_raw = config.get("CHANNELS_LIST", "")
    
    unique_channels = set()
    
    # 1. Сначала из .env
    if channels_raw:
        for c in channels_raw.split(","):
            c = c.strip()
            if not c: continue
            unique_channels.add(c)
            
    # 2. Теперь из БД те, что имеют статус ACTIVE
    try:
        with SessionLocal() as session:
            db_channels = session.query(ProcessingChannel).filter_by(status='ACTIVE').all()
            for db_c in db_channels:
                if db_c.url:
                    unique_channels.add(db_c.url)
                else:
                    unique_channels.add(f"@{db_c.username}")
    except Exception as e:
        logger.error(f"Error reading channels from DB: {e}")
            
    return sorted(list(unique_channels))

def get_disabled_channels():
    """Читает список отключенных каналов из .env.scraper."""
    from dotenv import dotenv_values
    config = dotenv_values(ENV_SCRAPER_PATH)
    channels_raw = config.get("DISABLED_CHANNELS_LIST", "")
    if not channels_raw:
        return []
    seen = set()
    unique = []
    for c in channels_raw.split(","):
        c = c.strip()
        if not c: continue
        can = canonical_source_name(c)
        if can not in seen:
            seen.add(can)
            unique.append(c)
    return unique

def update_env_channels(active: list, disabled: list):
    """Обновляет оба списка в .env.scraper."""
    set_key(ENV_SCRAPER_PATH, "CHANNELS_LIST", ",".join(active))
    set_key(ENV_SCRAPER_PATH, "DISABLED_CHANNELS_LIST", ",".join(disabled))

def get_sources_dashboard_data():
    """Сбор данных и формирование текста для основного дашборда источников."""
    active = get_active_channels()
    disabled = get_disabled_channels()
    
    text = "📂 **Управление источниками (Channels)**\n\n"
    
    if active:
        text += "✅ **Активные (мониторятся):**\n"
        for c in active:
            text += f"• {c}\n"
    else:
        text += "❌ Активных источников пока нет.\n"
        
    if disabled:
        text += "\n💤 **Отключенные:**\n"
        for d in disabled:
            text += f"• {d}\n"

    # --- Секция: Каналы в обработке ---
    try:
        with SessionLocal() as session:
            processing = session.query(ProcessingChannel).filter(
                ProcessingChannel.status.in_(['PENDING', 'SCRAPING', 'EXTRACTING'])
            ).all()
            
            if processing:
                text += "\n⏳ **Сейчас обрабатываются:**\n"
                status_icons = {
                    'PENDING': '📝 В очереди',
                    'SCRAPING': '📥 Сбор новостей',
                    'EXTRACTING': '🧠 ИИ-анализ (LLM)'
                }
                for p in processing:
                    icon = status_icons.get(p.status, p.status)
                    pbar = get_progress_bar(p.progress)
                    text += f"• @{p.username}\n  └ {icon} {pbar}\n"
                text += "\n_Анализ крупной истории может занять некоторое время. Мы сообщим о готовности!_\n"
    except Exception as e:
        logger.error(f"Error checking processing channels for dashboard: {e}")

    kb = get_sources_inline_keyboard(active, disabled)
    return text, kb

async def send_long_message(message: types.Message, text: str, parse_mode="HTML"):
    """Разбивает длинный текст и отправляет его частями (лимит TG 4096)."""
    # Если это ответ от LLM, санируем его (Markdown -> HTML)
    processed_text = sanitize_for_tg(text) if parse_mode == "HTML" else text
    
    max_length = 4000
    if len(processed_text) <= max_length:
        res = await safe_tg_call(message.answer(processed_text, parse_mode=parse_mode))
        if res == "ENTITY_ERROR":
            await message.answer(processed_text, parse_mode=None)
        return

    paragraphs = processed_text.split('\n')
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(paragraph) > max_length:
            if current_chunk.strip():
                res = await safe_tg_call(message.answer(current_chunk, parse_mode=parse_mode))
                if res == "ENTITY_ERROR": await message.answer(current_chunk)
                current_chunk = ""
            
            for i in range(0, len(paragraph), max_length):
                chunk = paragraph[i:i + max_length]
                res = await safe_tg_call(message.answer(chunk, parse_mode=parse_mode))
                if res == "ENTITY_ERROR": await message.answer(chunk)
            continue

        if len(current_chunk) + len(paragraph) + 1 <= max_length:
            current_chunk += paragraph + "\n"
        else:
            if current_chunk.strip():
                res = await safe_tg_call(message.answer(current_chunk, parse_mode=parse_mode))
                if res == "ENTITY_ERROR": await message.answer(current_chunk)
            current_chunk = paragraph + "\n"
            
    if current_chunk.strip():
        res = await safe_tg_call(message.answer(current_chunk, parse_mode=parse_mode))
        if res == "ENTITY_ERROR": await message.answer(current_chunk)

# Инициализация поисковых движков
basic_searcher = GraphRAGSearcher()
adv_searcher = AdvancedAnalyticalRouter()

# --- Текстовые константы ---
GREETING_TEXT = """
🚀 <b>Система аналитики GraphRAG активирована.</b>

Я — интеллектуальная платформа для глубокого анализа информационных потоков. Моя архитектура объединяет векторный поиск и графовые базы данных (GraphRAG), что позволяет мне не просто извлекать факты, но и выявлять неявные зависимости и скрытые тренды в ваших источниках данных.
"""

HELP_TEXT = """
📖 <b>Архитектура решений GraphRAG:</b>

📊 <b>Факт-чекинг (Режим 1)</b>: Быстрая экстракция фактов из базы знаний (кто, что, когда). Оптимально для верификации данных в режиме реального времени.

🕸 <b>Анализ связей (Режим 2)</b>: Построение многоуровневых причинно-следственных связей (Multi-hop reasoning). Выявляет косвенное влияние событий, компаний и макроэкономических факторов друг на друга.

📝 <b>Сводка новостей (Режим 3)</b>: Автоматизированный синтез разрозненных новостей в структурированный Executive Summary за выбранный период.

🌐 <b>Интеграция внешних данных (Fallback Search)</b>:
В случае отсутствия релевантной информации во внутренних источниках, системой предусмотрена возможность использовать веб-поиск. Внешние данные будут валидированы и интегрированы в итоговый отчет.

📡 <b>Управление контуром данных</b>:
Через меню «🔍 Выбор источников» вы можете динамически настраивать периметр мониторинга, подключая или исключая конкретные информационные каналы.
"""

# --- Хендлеры ---

@dp.message(Command("start"))
@dp.message(F.text == "🏠 О проекте")
@dp.message(F.text == "🏠 О проекте / Помощь")
@dp.message(F.text == "🏠 Главное меню")
@dp.message(F.text == "🏠 О боте / Режимы")
async def cmd_start(message: types.Message, state: FSMContext):
    """Только приветственный инфо-блок + кнопка перехода к режимам."""
    await state.clear()
    await message.answer(
        GREETING_TEXT,
        parse_mode="HTML",
        reply_markup=get_main_menu_keyboard()
    )
    # Предлагаем начать анализ отдельной кнопкой
    await message.answer(
        "Для начала работы выберите оптимальный аналитический режим:",
        reply_markup=get_start_analysis_keyboard()
    )

@dp.message(F.text == "ℹ️ Помощь")
@dp.message(F.text == "ℹ️ Справка")
async def cmd_help(message: types.Message, state: FSMContext):
    """Раздел детальной справки."""
    await state.clear()
    await message.answer(
        HELP_TEXT,
        parse_mode="HTML",
        reply_markup=get_main_menu_keyboard()
    )

@dp.callback_query(F.data == "start_analysis")
async def process_start_analysis(callback: types.CallbackQuery):
    """Показ выбора режимов после нажатия на кнопку-ракету."""
    await callback.answer()
    await callback.message.answer(
        "Выберите желаемый режим анализа:",
        reply_markup=get_modes_inline_keyboard()
    )

@dp.message(F.text == "🔍 Источники")
@dp.message(F.text == "🔍 Выбор источников")
async def cmd_sources(message: types.Message, state: FSMContext = None):
    text, kb = get_sources_dashboard_data()
    sent_msg = await message.answer(text, reply_markup=kb, parse_mode="Markdown")
    # Запоминаем ID сообщения для этого чата
    active_sources_dashboards[message.chat.id] = sent_msg.message_id

@dp.message(F.text == "💬 Задать вопрос")
@dp.message(F.text == "💬 Спросить бота")
async def cmd_ask_question(message: types.Message, state: FSMContext):
    """Теперь при нажатии 'Задать вопрос' мы всегда предлагаем выбор режима."""
    await state.clear()
    channels = get_active_channels()
    if not channels:
        await message.answer(
            "⚠️ *Внимание!* \n\nСписок источников пуст. Пожалуйста, сначала настройте каналы в разделе *Выбор источников*.",
            parse_mode="HTML",
            reply_markup=get_sources_inline_keyboard()
        )
        return

    await message.answer(
        "Выберите режим для вашего вопроса:",
        reply_markup=get_modes_inline_keyboard()
    )

# --- Обработка Inline-кнопок ---

@dp.callback_query(F.data.startswith("mode_"))
async def process_mode_select(callback: types.CallbackQuery, state: FSMContext):
    # Мгновенно отвечаем Telegram, чтобы кнопка не зависла и ID не просрочился
    try:
        await callback.answer()
    except Exception:
        pass

    channels = get_active_channels()
    
    if not channels:
        await callback.message.edit_text(
            "⚠️ *Внимание!* \n\nСписок источников пуст. Пожалуйста, сначала настройте каналы в разделе *Выбор источников*, иначе мне нечего будет анализировать.",
            parse_mode="HTML",
            reply_markup=get_sources_inline_keyboard()
        )
        return

    mode_id = callback.data.split("_")[1]
    await state.update_data(current_mode=mode_id)
    
    if mode_id == "3":
        await state.set_state(DigestState.waiting_for_topic)
        await callback.message.edit_text(
            "📅 *Режим Дайджеста (Главный Редактор)*\n\nЯ составлю для вас хронологическую сводку новостей.\n\n*Введите тему для дайджеста:* \n(Например: 'Газпром' или 'Санкции')",
            parse_mode="HTML",
            reply_markup=InlineKeyboardBuilder().row(
                types.InlineKeyboardButton(text="🏠 Отмена", callback_data="back_to_main")
            ).as_markup()
        )
    else:
        await state.set_state(AnalysisState.waiting_for_question)
        await callback.message.edit_text(
            f"🎯 *Выбран Режим {mode_id}*\n\nЯ готов проанализировать ваши данные. \n\n*Введите ваш вопрос боту:* \n(Например: 'Какие главные события произошли?' или 'Как санкции влияют на рынок?')",
            parse_mode="HTML",
            reply_markup=InlineKeyboardBuilder().row(
                types.InlineKeyboardButton(text="🏠 Отмена", callback_data="back_to_main")
            ).as_markup()
        )

@dp.message(AnalysisState.waiting_for_question, ~F.text.startswith("🏠"), ~F.text.startswith("🔍"), ~F.text.startswith("💬"), ~F.text.startswith("ℹ️"))
async def handle_analysis_question(message: types.Message, state: FSMContext):
    """Обработка вопроса пользователя в режиме анализа."""
    query = message.text.strip()
    if len(query) < 5:
        await message.answer("❌ Вопрос слишком короткий. Попробуйте сформулировать его подробнее.")
        return

    data = await state.get_data()
    mode_id = data.get("current_mode", "1")
    
    await state.update_data(gap_query=query, gap_mode=mode_id)
    
    status_text = "🔍 <b>Исполняю базовый поиск...</b>" if mode_id == "1" else "🧠 <b>Запускаю глубокий аналитический поиск...</b> \nЭто может занять некоторое время.."
    log_history = [status_text]
    status_msg = await message.answer(status_text, parse_mode="HTML")
    
    async def on_status_update(new_status: str):
        log_history.append(f"✅ {new_status}")
        full_log = "\n".join(log_history)
        if len(full_log) > 500: full_log = "..." + full_log[-500:]
        try:
            await safe_tg_call(status_msg.edit_text(full_log, parse_mode="HTML"))
        except Exception:
            await safe_tg_call(status_msg.edit_text(full_log))

    current_searcher = adv_searcher if mode_id == "2" else basic_searcher
    
    try:
        answer_gen = current_searcher.ask(query, on_status=on_status_update)
        full_log = "\n".join(log_history)
        prefix = f"{full_log}\n\n📊 <b>РЕЗУЛЬТАТ (Режим {mode_id}):</b>\n\n"
        accumulated_answer = ""
        last_update_time = time.time()
        
        async for chunk in answer_gen:
            if chunk == SIGNAL_GAP_DETECTED:
                await status_msg.edit_text(
                    f"🧐 *Данных в локальной базе недостаточно!*\n\nЯ не нашел актуальной информации по запросу '{query}' в подключенных каналах.\n\nХотите, чтобы я поискал в глобальном интернете?",
                    parse_mode="HTML",
                    reply_markup=get_web_fallback_keyboard(query, mode_id)
                )
                return

            accumulated_answer += chunk
            if time.time() - last_update_time > 1.5:
                display_text = prefix + accumulated_answer
                if len(display_text) > 2800:
                    display_text = display_text[:2700] + "\n\n⚠️ <b>[Текст слишком длинный...]</b>"
                try:
                    await safe_tg_call(status_msg.edit_text(display_text, parse_mode="HTML"))
                except Exception:
                    await safe_tg_call(status_msg.edit_text(display_text))
                last_update_time = time.time()
        
        final_text = f"{full_log}\n\n📊 <b>РЕЗУЛЬТАТ (Режим {mode_id}):</b>\n\n{accumulated_answer}"
        await send_long_message(message, final_text)
        try: await status_msg.delete()
        except: pass
                
    except Exception as e:
        logger.error(f"Error in query (Mode {mode_id}): {e}")
        await message.answer(f"❌ <b>Ошибка при поиске:</b> \n<code>{html.escape(str(e))}</code>", parse_mode="HTML")
    
    await message.answer(
        "Желаете спросить что-то еще или дополнить ответ через интернет?", 
        reply_markup=get_modes_inline_keyboard(show_web_search=True)
    )
    await state.set_state(AnalysisState.waiting_for_question)

@dp.message(DigestState.waiting_for_topic)

@dp.message(DigestState.waiting_for_topic)
async def process_digest_topic(message: types.Message, state: FSMContext):
    """Шаг 1 Режима 3: Принимаем тему."""
    topic = message.text.strip()
    if len(topic) < 3:
        await message.answer("❌ Слишком короткая тема. Попробуйте еще раз.")
        return
    await state.update_data(digest_topic=topic)
    await state.set_state(DigestState.waiting_for_date)
    await message.answer(
        f"📝 Тема: <b>{html.escape(topic)}</b>\n\nТеперь введите дату или диапазон в формате:\n<code>ДД.ММ.ГГГГ</code> (например, <code>17.04.2026</code>)\nили\n<code>ДД.ММ.ГГГГ-ДД.ММ.ГГГГ</code> (например, <code>15.04.2026-17.04.2026</code>)",
        parse_mode="HTML"
    )

@dp.message(DigestState.waiting_for_date)
async def process_digest_date(message: types.Message, state: FSMContext):
    """Шаг 2 Режима 3: Принимаем дату и запускаем генерацию."""
    date_input = message.text.strip()
    if date_input.lower() in ["отмена", "назад", "🏠 главное меню"]:
        await state.clear()
        await cmd_start(message, state)
        return

    try:
        adv_searcher._parse_dates(date_input)
    except ValueError as e:
        min_d, max_d = await adv_searcher._get_available_date_range()
        range_msg = f"\n\nУ меня есть новости за период: **{min_d} — {max_d}**" if min_d else ""
        await message.answer(
            f"⚠️ <b>Ошибка формата даты:</b>\n{html.escape(str(e))}{range_msg}\n\nПожалуйста, введите дату корректно (например, <code>21.04.2026</code>) или введите <code>отмена</code>.",
            parse_mode="HTML"
        )
        return

    data = await state.get_data()
    topic = data.get("digest_topic")
    await state.update_data(gap_query=topic, gap_mode="3", gap_date=date_input)

    status_text = "🗞 <b>Начинаю работу над дайджестом...</b>"
    log_history = [status_text]
    status_msg = await message.answer(status_text, parse_mode="HTML")

    async def on_status_update(new_status: str):
        log_history.append(f"✅ {new_status}")
        full_log = "\n".join(log_history)
        if len(full_log) > 500: full_log = "..." + full_log[-500:]
        try: await status_msg.edit_text(full_log, parse_mode="HTML")
        except Exception: await status_msg.edit_text(full_log)

    try:
        answer_gen = adv_searcher.generate_digest(topic, date_input, on_status=on_status_update)
        full_log = "\n".join(log_history)
        prefix = f"{full_log}\n\n"
        accumulated_answer = ""
        last_update_time = time.time()
        is_error = False
        
        async for chunk in answer_gen:
            if chunk == SIGNAL_GAP_DETECTED:
                await status_msg.edit_text(
                    f"🧐 *В архивах за этот период пусто!*\n\nПо теме '{topic}' за указанные даты новостей не найдено.\n\nПопробовать поискать в интернете?",
                    parse_mode="HTML",
                    reply_markup=get_web_fallback_keyboard(topic, "3")
                )
                return
            if chunk.startswith("⚠️"): is_error = True
            accumulated_answer += chunk
            if time.time() - last_update_time > 1.5:
                display_text = prefix + sanitize_for_tg(accumulated_answer)
                if len(display_text) > 2800: display_text = display_text[:2700] + "\n\n⚠️ <b>[Текст слишком длинный...]</b>"
                try: await safe_tg_call(status_msg.edit_text(display_text, parse_mode="HTML"))
                except Exception: await safe_tg_call(status_msg.edit_text(display_text))
                last_update_time = time.time()
        
        if is_error:
            await message.answer(f"{accumulated_answer}\n\nПожалуйста, введите корректную дату или диапазон:", parse_mode="HTML")
            try: await status_msg.delete()
            except: pass
            return

        final_text = f"{full_log}\n\n{accumulated_answer}"
        await send_long_message(message, final_text)
        try: await status_msg.delete()
        except: pass
        await state.set_state(AnalysisState.waiting_for_question)
        await message.answer("Сводка готова!", reply_markup=get_modes_inline_keyboard(show_web_search=True))
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error in Digest Mode: {e}")
        await message.answer(f"❌ <b>Ошибка при создании дайджеста:</b> \n<code>{html.escape(str(e))}</code>", parse_mode="HTML")

@dp.callback_query(F.data == "web_search_manual")
async def handle_web_search_manual(callback: types.CallbackQuery, state: FSMContext):
    """Обработка ручного нажатия кнопки поиска в интернете (после ответа)."""
    data = await state.get_data()
    query = data.get("gap_query")
    mode_id = data.get("current_mode", "1")
    
    if not query:
        await callback.answer("❌ Не удалось определить тему запроса для поиска.")
        return
        
    await _perform_web_search_logic(callback, state, query, mode_id)

@dp.callback_query(F.data.startswith("web_search:"))
async def handle_web_search_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработка автоматической кнопки веб-поиска (при детекции пропуска)."""
    mode_id = callback.data.split(":")[1]
    data = await state.get_data()
    query = data.get("gap_query")
    
    await _perform_web_search_logic(callback, state, query, mode_id)

async def _perform_web_search_logic(callback: types.CallbackQuery, state: FSMContext, query: str, mode_id: str):
    """Общая внутренняя логика выполнения поиска и ре-синтеза."""
    try:
        await callback.answer("🌐 Запускаю глобальный поиск...")
    except: pass

    data = await state.get_data()
    date_input = data.get("gap_date") # Только для режима 3

    status_msg = callback.message
    log_history = ["🌐 *Запускаю Tavily Web Search...*"]
    await status_msg.edit_text("\n".join(log_history), parse_mode="Markdown")

    current_searcher = adv_searcher if mode_id in ["2", "3"] else basic_searcher
    
    try:
        # 1. Выполняем веб-поиск
        print(f"🤖 [BOT] Initiating web fallback for: {query}")
        search_depth = "advanced" if mode_id == "2" else "basic"
        web_raw_results = await current_searcher.web_search(query, depth=search_depth)
        
        if "⚠️ Ошибка" in web_raw_results:
             await status_msg.edit_text(web_raw_results)
             return

        log_history.append("✅ Данные из интернета получены")
        log_history.append("🤖 Пересчитываю аналитический ответ...")
        await status_msg.edit_text("\n".join(log_history), parse_mode="Markdown")
        print(f"🤖 [BOT] Web context received, starting re-synthesis...")

        # 2. Запускаем повторный синтез с учетом Web-контекста
        if mode_id == "3":
            answer_gen = current_searcher.generate_digest(query, date_input, web_context=web_raw_results)
        else:
            answer_gen = current_searcher.ask(query, web_context=web_raw_results)

        prefix = "\n".join(log_history) + "\n\n🌐 *ОТВЕТ НА ОСНОВЕ WEB-ДАННЫХ:*\n\n"
        accumulated_answer = ""
        last_update_time = time.time()

        async for chunk in answer_gen:
            accumulated_answer += chunk
            if time.time() - last_update_time > 1.5:
                try: 
                    await safe_tg_call(status_msg.edit_text(prefix + accumulated_answer, parse_mode="Markdown"))
                except: 
                    await safe_tg_call(status_msg.edit_text(prefix + accumulated_answer))
                last_update_time = time.time()

        # Финальная отправка
        final_text = f"🌐 *ИСПОЛЬЗОВАН WEB-ПОИСК (Tavily)*\n\n{accumulated_answer}"
        await send_long_message(callback.message, final_text)
        try: await status_msg.delete()
        except: pass

    except Exception as e:
        logger.error(f"Error in Web Search: {e}")
        await callback.message.answer(f"❌ *Ошибка веб-поиска:* \n`{str(e)}`", parse_mode="Markdown")

@dp.callback_query(F.data == "back_to_main")
async def process_back_main(callback: types.CallbackQuery):
    await callback.message.edit_text(
        "🏠 Вы вернулись в выбор режимов:",
        reply_markup=get_modes_inline_keyboard()
    )
    await callback.answer()

# --- Управление источниками (Хендлеры) ---

async def refresh_sources_dashboard(message: types.Message):
    """Вспомогательная функция для обновления дашборда при нажатии кнопок."""
    text, kb = get_sources_dashboard_data()
    try:
        await message.edit_text(text, parse_mode="Markdown", reply_markup=kb)
        # Если это было новое сообщение (не из дашборда), обновим трекер
        active_sources_dashboards[message.chat.id] = message.message_id
    except Exception as e:
        if "message is not modified" not in str(e).lower():
            logger.error(f"Error refreshing sources dashboard: {e}")

@dp.callback_query(F.data.startswith("toggle_act_"))
async def process_toggle_active(callback: types.CallbackQuery):
    """Отключение активного источника (перенос в disabled)."""
    index = int(callback.data.split("_")[2])
    active = get_active_channels()
    disabled = get_disabled_channels()
    
    if 0 <= index < len(active):
        removed = active.pop(index)
        disabled.append(removed)
        update_env_channels(active, disabled)
        await callback.answer(f"💤 Отключено: {removed.split('/')[-1]}")
        await refresh_sources_dashboard(callback.message)
    else:
        await callback.answer("Ошибка: источник не найден.")

@dp.callback_query(F.data.startswith("toggle_dis_"))
async def process_toggle_disabled(callback: types.CallbackQuery):
    """Включение отключенного источника (перенос в active)."""
    index = int(callback.data.split("_")[2])
    active = get_active_channels()
    disabled = get_disabled_channels()
    
    if 0 <= index < len(disabled):
        restored = disabled.pop(index)
        active.append(restored)
        update_env_channels(active, disabled)
        await callback.answer(f"✅ Включено: {restored.split('/')[-1]}")
        await refresh_sources_dashboard(callback.message)
    else:
        await callback.answer("Ошибка: источник не найден.")

@dp.callback_query(F.data == "add_source")
async def process_start_add(callback: types.CallbackQuery, state: FSMContext):
    try:
        await callback.message.edit_text(
            "➕ *Добавление источника*\n\nОтправьте ссылку на канал (например, `t.me/example`).",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardBuilder().row(
                types.InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_main")
            ).as_markup()
        )
        await state.set_state(SourceManagement.waiting_for_source_url)
    except Exception as e:
        if "message is not modified" not in str(e).lower():
            logger.error(f"Error in process_start_add: {e}")
            
    try:
        await callback.answer()
    except Exception:
        pass

@dp.message(SourceManagement.waiting_for_source_url)
async def handle_add_source(message: types.Message, state: FSMContext):
    new_url = message.text.strip()
    if not new_url.startswith(("t.me/", "http", "@")):
        await message.answer("❌ Неверный формат ссылки. Пришлите ссылку вида t.me/channel или @username.")
        return

    username = canonical_source_name(new_url)
    if not username:
        await message.answer("❌ Не удалось распознать имя канала из ссылки.")
        return

    # 1. Проверяем, есть ли уже такой канал
    with SessionLocal() as session:
        exists = session.query(ProcessingChannel).filter_by(username=username).first()
        if exists:
            status_map = {
                'PENDING': 'в очереди на запуск',
                'SCRAPING': 'в процессе сбора новостей',
                'EXTRACTING': 'в процессе анализа (LLM)',
                'ACTIVE': 'уже активен и доступен для поиска',
                'ERROR': 'завершен с ошибкой'
            }
            curr_status = status_map.get(exists.status, exists.status)
            await message.answer(f"⚠️ Канал @{username} уже зарегистрирован в системе.\n\nТекущий статус: **{curr_status}**")
            await state.clear()
            await cmd_sources(message, state)
            return

        # 2. Создаем запись в БД
        new_channel = ProcessingChannel(
            username=username,
            url=new_url,
            status='PENDING',
            progress=0,
            last_chat_id=message.chat.id
        )
        session.add(new_channel)
        session.commit()
        # Сохраняем ID канала для обновления после отправки сообщения
        db_channel_id = new_channel.id

    # 3. Триггерим Airflow
    try:
        dag_id = "process_new_channel"
        url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns"
        payload = {
            "conf": {"channel_username": username, "channel_url": new_url}
        }
        
        # Вызов API Airflow (внутренняя сеть Docker)
        response = requests.post(url, json=payload, auth=AIRFLOW_AUTH, timeout=10)
        
        if response.status_code == 200 or response.status_code == 201:
            sent_msg = await message.answer(
                f"✅ Канал @{username} принят в обработку!\n\n"
                "🚀 Мы запускаем автоматизированный конвейер:\n"
                "1. Сбор последних новостей за 30 дней\n"
                "2. Экстракция сущностей через LLM (Gemini)\n"
                "3. Построение графа связей\n\n"
                "⏳ **Статус:** 📝 В очереди [░░░░░░░░░░] 0%\n\n"
                "⚠️ _Анализ истории может занять от 2 до 10 минут._",
                parse_mode="Markdown"
            )
            # Обновляем message_id в БД
            with SessionLocal() as session:
                session.execute(
                    update(ProcessingChannel)
                    .where(ProcessingChannel.id == db_channel_id)
                    .values(last_message_id=sent_msg.message_id)
                )
                session.commit()
        else:
            logger.error(f"Airflow API Error: {response.status_code} - {response.text}")
            await message.answer("⚠️ Канал добавлен в БД, но не удалось автоматически запустить конвейер. Администратор проверит систему.")
            
    except Exception as e:
        logger.error(f"Error triggering Airflow: {e}")
        await message.answer("⚠️ Произошла техническая ошибка при запуске конвейера Airflow. Канал будет обработан позже автоматически.")
    
    await state.clear()
    await cmd_sources(message, state)

# --- Фоновая задача мониторинга прогресса ---

async def monitor_processing():
    """Фоновая задача для обновления прогресс-баров в сообщениях."""
    logger.info("Task: Progress Monitoring started.")
    
    last_status_cache = {} # Храним последнее состояние, чтобы не спамить Edit API

    while True:
        try:
            with SessionLocal() as session:
                # Берем все каналы, которые еще не завершились или недавно завершились
                channels = session.query(ProcessingChannel).filter(
                    ProcessingChannel.status.in_(['PENDING', 'SCRAPING', 'EXTRACTING', 'ACTIVE', 'ERROR'])
                ).all()

                for ch in channels:
                    if not ch.last_chat_id or not ch.last_message_id:
                        continue
                        
                    # Ключ для кеша
                    cache_key = f"{ch.username}_{ch.status}_{ch.progress}"
                    if cache_key in last_status_cache:
                        # Если статус ACTIVE/ERROR и мы его уже отработали - пропускаем
                        if ch.status in ['ACTIVE', 'ERROR']:
                            continue
                        # Если прогресс не изменился - пропускаем
                        if last_status_cache[cache_key] == ch.progress:
                            continue

                    status_icons = {
                        'PENDING': '📝 В очереди',
                        'SCRAPING': '📥 Сбор новостей',
                        'EXTRACTING': '🧠 ИИ-анализ (LLM)',
                        'ACTIVE': '✅ Готово',
                        'ERROR': '❌ Ошибка'
                    }
                    
                    icon = status_icons.get(ch.status, ch.status)
                    pbar = get_progress_bar(ch.progress)
                    
                    text = (
                        f"📡 **Обновление статуса: @{ch.username}**\n\n"
                        f"🚀 Конвейер обработки запущен:\n"
                        f"1. Сбор последних новостей\n"
                        f"2. Экстракция сущностей (LLM)\n"
                        f"3. Построение графа\n\n"
                        f"⏳ **Статус:** {icon} {pbar}"
                    )
                    
                    if ch.status == 'ACTIVE':
                        text = (
                            f"🎉 **Канал @{ch.username} полностью обработан!**\n\n"
                            "✅ Данные успешно импортированы в GraphRAG.\n"
                            "Теперь вы можете задавать вопросы по информации из этого источника.\n\n"
                            "🏁 [▓▓▓▓▓▓▓▓▓▓] 100%"
                        )
                    elif ch.status == 'ERROR':
                        text = (
                            f"❌ **Ошибка при обработке канала @{ch.username}**\n\n"
                            f"Причина: `{ch.error_message or 'Неизвестная ошибка'}`\n\n"
                            "⚠️ Попробуйте добавить канал позже ."
                        )

                    try:
                        try:
                            await safe_tg_call(bot.edit_message_text(
                                text=text,
                                chat_id=ch.last_chat_id,
                                message_id=ch.last_message_id,
                                parse_mode="Markdown"
                            ))
                        except Exception:
                            await safe_tg_call(bot.edit_message_text(
                                text=text,
                                chat_id=ch.last_chat_id,
                                message_id=ch.last_message_id
                            ))
                        last_status_cache[cache_key] = ch.progress
                        
                        # Если завершили - удаляем из кеша старые записи этого канала и метим как отработанное
                        if ch.status in ['ACTIVE', 'ERROR']:
                            # Очистка кеша для этого канала (упрощенно)
                            pass
                            
                    except Exception as e:
                        if "message is not modified" not in str(e):
                            logger.error(f"Error editing progress message for @{ch.username}: {e}")
                        last_status_cache[cache_key] = ch.progress

                # --- 2. ОБНОВЛЕНИЕ ГЛАВНЫХ ДАШБОРДОВ ---
                if active_sources_dashboards:
                    dash_text, dash_kb = get_sources_dashboard_data()
                    # Хешируем текст, чтобы не дергать API зря
                    dash_hash = hash(dash_text)
                    
                    for chat_id, msg_id in list(active_sources_dashboards.items()):
                        # Если в этом чате открыт дашборд и его текст изменился
                        if last_status_cache.get(f"dash_{chat_id}") != dash_hash:
                            try:
                                await bot.edit_message_text(
                                    text=dash_text,
                                    chat_id=chat_id,
                                    message_id=msg_id,
                                    reply_markup=dash_kb,
                                    parse_mode="Markdown"
                                )
                                last_status_cache[f"dash_{chat_id}"] = dash_hash
                            except Exception as e:
                                if "message is not modified" in str(e).lower():
                                    last_status_cache[f"dash_{chat_id}"] = dash_hash
                                elif "message to edit not found" in str(e).lower() or "chat not found" in str(e).lower():
                                    # Удаляем из трекера, если сообщение удалено
                                    active_sources_dashboards.pop(chat_id, None)
                                else:
                                    logger.error(f"Error auto-updating dashboard for {chat_id}: {e}")

        except Exception as e:
            logger.error(f"Error in monitor_processing loop: {e}")
            
        await asyncio.sleep(4) # Опрашиваем раз в 4 секунды

async def main():
    global bot
    if not BOT_TOKEN:
        logger.error("TG_BOT_TOKEN не найден в .env.db. Пожалуйста, добавьте его!")
        return
    
    # Инициализация бота через стабильный HTTP-прокси (порт 2081)
    session = AiohttpSession(proxy="http://finland_proxy:2081")
    bot = Bot(token=BOT_TOKEN, session=session)
    
    logger.info("Бот запущен...")
    
    # Запуск сервера метрик Prometheus (порт 8000)
    telemetry.start_server(8000)
    
    # Запускаем фоновый мониторинг
    asyncio.create_task(monitor_processing())
    
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА ПРИ ОПРОСЕ (polling): {e}", exc_info=True)
    finally:
        await bot.session.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен")