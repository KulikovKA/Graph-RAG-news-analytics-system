import re
from typing import Union, List, Dict, Any

# Глобальный экземпляр Mystem (ленивая загрузка)
_mystem = None

def get_mystem():
    global _mystem
    if _mystem is None:
        try:
            from pymystem3 import Mystem
            _mystem = Mystem()
        except ImportError:
            return None
    return _mystem

def normalize_text(text: str) -> str:
    """Лемматизация: ед. число, именительный падеж, очистка."""
    if not text:
        return ""
    text = text.replace("_", " ").strip()
    
    mystem_instance = get_mystem()
    if mystem_instance:
        lemmas = mystem_instance.lemmatize(text.lower())
        clean_lemmas = [l.strip() for l in lemmas if l.strip()]
        normalized = " ".join(clean_lemmas)
        return normalized.title().replace(" ", "_")
    
    # Fallback если mystem недоступен
    return text.title().replace(" ", "_")

def ensure_string(content: Union[str, List[Any]]) -> str:
    """Гарантирует, что контент от ИИ является строкой (обрабатывает List[Part])."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Собираем текстовые части из списка (формат Gemini/LangChain)
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)
