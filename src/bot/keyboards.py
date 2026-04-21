from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder

def get_main_menu_keyboard():
    """Главное меню (Reply Keyboard, всегда доступно)."""
    builder = ReplyKeyboardBuilder()
    builder.row(
        KeyboardButton(text="🏠 О проекте"),
        KeyboardButton(text="ℹ️ Помощь")
    )
    builder.row(KeyboardButton(text="💬 Задать вопрос"))
    builder.row(
        KeyboardButton(text="🔍 Источники")
    )
    return builder.as_markup(resize_keyboard=True, persistent=True)

def get_start_analysis_keyboard():
    """Кнопка для перехода к выбору режимов."""
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="🚀 Перейти к выбору режима", callback_data="start_analysis"))
    return builder.as_markup()

def get_modes_inline_keyboard(show_web_search: bool = False):
    """Выбор режимов анализа (Inline)."""
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="📊 Факт-чекинг (Режим 1)", callback_data="mode_1"))
    builder.row(InlineKeyboardButton(text="🕸 Анализ связей (Режим 2)", callback_data="mode_2"))
    builder.row(InlineKeyboardButton(text="📝 Сводка новостей (Режим 3)", callback_data="mode_3"))
    
    if show_web_search:
        builder.row(InlineKeyboardButton(text="🌐 Искать этот запрос в интернете", callback_data="web_search_manual"))
        
    return builder.as_markup()

def get_sources_inline_keyboard(active=None, disabled=None):
    """Управление источниками: Дашборд с чекбоксами."""
    builder = InlineKeyboardBuilder()
    
    # 1. Активные источники
    if active:
        for i, channel in enumerate(active):
            short_name = channel.split('/')[-1]
            builder.row(InlineKeyboardButton(text=f"✅ {short_name}", callback_data=f"toggle_act_{i}"))
    
    # 2. Отключенные источники (то, что просил юзер: название + крестик)
    if disabled:
        for i, channel in enumerate(disabled):
            short_name = channel.split('/')[-1]
            builder.row(InlineKeyboardButton(text=f"❌ {short_name}", callback_data=f"toggle_dis_{i}"))
    
    builder.row(InlineKeyboardButton(text="➕ Добавить новый", callback_data="add_source"))
    return builder.as_markup()

def get_delete_sources_keyboard(channels):
    """Клавиатура для удаления конкретных источников."""
    builder = InlineKeyboardBuilder()
    for i, channel in enumerate(channels):
        # Очищаем название для кнопки (берем конец ссылки)
        short_name = channel.split('/')[-1]
        builder.row(InlineKeyboardButton(text=f"❌ {short_name}", callback_data=f"del_{i}"))
    
    builder.row(InlineKeyboardButton(text="⬅️ Отмена", callback_data="list_sources"))
    return builder.as_markup()

def get_web_fallback_keyboard(query: str, mode: str, date_input: str = None):
    """Клавиатура с предложением выполнить поиск в интернете."""
    builder = InlineKeyboardBuilder()
    # Кодируем данные для колбэка. Ограничение 64 байта, поэтому используем короткие префиксы.
    # Если запрос слишком длинный, мы сохраним его в FSM, а здесь передадим только сигнал.
    builder.row(InlineKeyboardButton(
        text="🌐 Искать в интернете (Tavily)", 
        callback_data=f"web_search:{mode}"
    ))
    builder.row(InlineKeyboardButton(text="🏠 В главное меню", callback_data="back_to_main"))
    return builder.as_markup()
