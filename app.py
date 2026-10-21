import io
import os
import re
import json
import hashlib
import urllib.request
import streamlit as st
from dotenv import load_dotenv

# Подгружаем переменные окружения (.env)
load_dotenv()

# --- БИБЛИОТЕКИ ПАРСИНГА И ИИ ---
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None


# --- КОНФИГУРАЦИЯ ВСЕХ ИИ-ПРОВАЙДЕРОВ ---

PROVIDERS = {
    "⚡ Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "placeholder": "gsk_...",
        "default_model": "openai/gpt-oss-120b",
        "needs_key": True,
    },
    "🌐 OpenRouter (Claude / GPT / DeepSeek / Free)": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "placeholder": "sk-or-v1-...",
        "default_model": "deepseek/deepseek-chat:free",
        "needs_key": True,
    },
    "🧠 OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "placeholder": "sk-...",
        "default_model": "gpt-4o-mini",
        "needs_key": True,
    },
    "🐋 DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "placeholder": "sk-...",
        "default_model": "deepseek-chat",
        "needs_key": True,
    },
    "✨ Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GEMINI_API_KEY",
        "placeholder": "AIzaSy...",
        "default_model": "gemini-2.5-flash",
        "needs_key": True,
    },
    "🛠️ Свой API (Mistral / xAI / Together / LM Studio)": {
        "base_url": os.getenv("CUSTOM_BASE_URL", "https://api.mistral.ai/v1"),
        "env_key": "CUSTOM_API_KEY",
        "placeholder": "Ваш API ключ...",
        "default_model": "mistral-large-latest",
        "needs_key": True,
        "custom_url": True,
    },
    "♾️ Ollama (Локально)": {
        "base_url": "http://localhost:11434/v1",
        "env_key": "",
        "placeholder": "",
        "default_model": "qwen2.5:14b",
        "needs_key": False,
    },
}


# --- УТИЛИТЫ ИЗВЛЕЧЕНИЯ КОНТЕНТА ---

def extract_youtube_id(url: str) -> str | None:
    """Извлекает 11-символьный ID видео из любых ссылок YouTube (youtu.be, shorts, watch, live)"""
    url = url.strip()
    patterns = [
        r"(?:v=|\/embed\/|\/live\/|\/shorts\/|\/v\/)([0-9A-Za-z_-]{11})(?:[\?&\/]|$)",
        r"youtu\.be\/([0-9A-Za-z_-]{11})(?:[\?&\/]|$)",
        r"^([0-9A-Za-z_-]{11})$"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _parse_transcript_data(data) -> str:
    """Универсальный парсер для любых версий youtube-transcript-api (v0.6+ и v1.0+)"""
    if hasattr(data, "to_raw_data"):
        data = data.to_raw_data()
    chunks = []
    for item in data:
        if isinstance(item, dict) and "text" in item:
            chunks.append(item["text"])
        elif hasattr(item, "text"):
            chunks.append(str(getattr(item, "text")))
    return " ".join(chunks).strip()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_youtube_transcript(url: str) -> str:
    """Надёжное извлечение субтитров YouTube (поддержка API v1.x, v0.x + фолбэк yt-dlp)"""
    if not YouTubeTranscriptApi:
        raise ImportError("Библиотека не установлена. Выполните: pip install -U youtube-transcript-api")
    
    vid = extract_youtube_id(url)
    if not vid:
        raise ValueError(f"Не удалось распознать ID видео из ссылки: {url}")

    errors_log = []

    # 1. Новый синтаксис youtube-transcript-api (v1.0.0+)
    try:
        ytt_api = YouTubeTranscriptApi()
        if hasattr(ytt_api, "list"):
            transcript_list = ytt_api.list(vid)
            try:
                transcript = transcript_list.find_transcript(["ru", "en", "uk", "be"])
            except Exception:
                transcript = next(iter(transcript_list))
            
            data = transcript.fetch()
            text = _parse_transcript_data(data)
            if text:
                return text
        elif hasattr(ytt_api, "fetch"):
            data = ytt_api.fetch(vid, languages=["ru", "en"])
            text = _parse_transcript_data(data)
            if text:
                return text
    except Exception as e:
        errors_log.append(f"API v1: {e}")

    # 2. Старый синтаксис youtube-transcript-api (v0.6.x)
    try:
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(vid)
            try:
                transcript = transcript_list.find_transcript(["ru", "en"])
            except Exception:
                transcript = next(iter(transcript_list))
            data = transcript.fetch()
            text = _parse_transcript_data(data)
            if text:
                return text
        elif hasattr(YouTubeTranscriptApi, "get_transcript"):
            data = YouTubeTranscriptApi.get_transcript(vid, languages=["ru", "en"])
            text = _parse_transcript_data(data)
            if text:
                return text
    except Exception as e:
        errors_log.append(f"API v0.6: {e}")

    # 3. Резервный метод через yt-dlp (если установлен в окружении)
    try:
        import yt_dlp
        ydl_opts = {"skip_download": True, "quiet": True, "writesubtitles": True, "writeautomaticsub": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
            subs = info.get("subtitles") or {}
            auto_subs = info.get("automatic_captions") or {}
            all_subs = {**auto_subs, **subs}

            lang = next((l for l in ["ru", "en"] if l in all_subs), next(iter(all_subs), None))
            if lang:
                formats = all_subs[lang]
                json3_url = next((f["url"] for f in formats if f.get("ext") == "json3"), formats[0]["url"])
                req = urllib.request.Request(json3_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    sub_data = json.loads(resp.read().decode("utf-8"))
                    words = [
                        seg["utf8"].strip()
                        for ev in sub_data.get("events", [])
                        for seg in ev.get("segs", [])
                        if "utf8" in seg and seg["utf8"].strip()
                    ]
                    text = " ".join(words)
                    if text:
                        return text
    except ImportError:
        pass
    except Exception as e:
        errors_log.append(f"yt-dlp: {e}")

    details = " | ".join(errors_log) if errors_log else "Субтитры не найдены"
    raise ValueError(f"Не удалось извлечь субтитры для видео `{vid}`.\nПодробности: {details}")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_web_article(url: str) -> str:
    """Кэшируемое извлечение текста статьи по URL"""
    if not trafilatura:
        raise ImportError("Библиотека не установлена. Выполните: pip install trafilatura")
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError("Не удалось загрузить веб-страницу (возможно, сайт защищен Cloudflare).")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    if not text:
        raise ValueError("Не удалось извлечь читаемый текст со страницы.")
    return text


@st.cache_data(show_spinner=False)
def fetch_pdf_text(file_bytes: bytes) -> str:
    """Кэшируемое чтение PDF из байтов"""
    if not PdfReader:
        raise ImportError("Библиотека не установлена. Выполните: pip install pypdf")
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, p in enumerate(reader.pages):
        t = p.extract_text()
        if t and t.strip():
            pages.append(f"--- Страница {i+1} ---\n{t.strip()}")
    
    result = "\n\n".join(pages)
    if not result:
        raise ValueError("В PDF не найден текстовый слой (возможно, это скан без OCR).")
    return result


def clean_token(token: str) -> str:
    """Удаляет пробелы, кавычки и не-ASCII символы из API-ключа"""
    if not token:
        return ""
    return "".join(c for c in token.strip().replace('"', '').replace("'", "") if ord(c) < 128)


# --- ДИНАМИЧЕСКАЯ ЗАГРУЗКА И РАНЖИРОВАНИЕ МОДЕЛЕЙ ДЛЯ ЛЮБОГО ИИ ---

PRIORITY_PATTERNS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "gpt-oss",
    "claude-3-7",
    "claude-3.5",
    "gpt-4o",
    "gemini-2.5",
    "gemini-2.0",
    "qwen3",
    "qwen-2.5-72b",
    "deepseek-r1",
    "deepseek-chat",
    "deepseek",
    "qwen",
    "llama-3.3-70b",
    "llama-3.1-70b",
    "70b",
    "120b",
    "32b",
    "27b",
]


def rank_models(model_ids: list[str]) -> list[str]:
    """Сортирует доступные модели так, чтобы самые сильные шли первыми"""
    def score(m_id: str) -> tuple[int, str]:
        m_low = m_id.lower()
        for idx, pattern in enumerate(PRIORITY_PATTERNS):
            if pattern in m_low:
                return (idx, m_low)
        return (len(PRIORITY_PATTERNS), m_low)

    return sorted(model_ids, key=score)


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_fetch_models(base_url: str, key_hash: str, _raw_key: str) -> tuple[list[str], str | None]:
    """
    Динамически запрашивает список моделей у любого OpenAI-совместимого провайдера.
    Кэшируется по хэшу ключа (_raw_key с нижним подчеркиванием не сохраняется в открытом кэше).
    """
    if not OpenAI:
        return [], "Библиотека openai не установлена (pip install openai)"

    try:
        client = OpenAI(base_url=base_url, api_key=_raw_key or "ollama")
        data = client.models.list().data
        
        exclude_keywords = (
            "whisper", "guard", "tts", "playai", "orpheus", 
            "embed", "dall-e", "moderation", "audio", "realtime", "babbage", "davinci"
        )
        text_models = [
            m.id for m in data
            if not any(ex in m.id.lower() for ex in exclude_keywords)
        ]
        
        if not text_models:
            text_models = [m.id for m in data]
            
        return rank_models(text_models), None
    except Exception as e:
        return [], str(e)


def get_provider_models(base_url: str, api_key: str, needs_key: bool) -> tuple[list[str], str | None]:
    """Безопасная обёртка для получения списка моделей"""
    clean_key = clean_token(api_key)
    if needs_key and not clean_key:
        return [], "Требуется API-ключ"
    key_hash = hashlib.sha256(clean_key.encode()).hexdigest()[:16] if clean_key else "local"
    return _cached_fetch_models(base_url, key_hash, clean_key)


def stream_llm(prompt: str, system: str, base_url: str, api_key: str, model_name: str, temperature: float = 0.3):
    """Универсальный генератор потока ответа для любого ИИ-провайдера"""
    if not OpenAI:
        raise ImportError("Библиотека openai не установлена. Выполните: pip install openai")
    
    client = OpenAI(base_url=base_url, api_key=api_key or "ollama")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        stream=True,
        temperature=temperature
    )
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# --- ИНТЕРФЕЙС STREAMLIT ---

st.set_page_config(page_title="Omni-Extractor AI", page_icon="⚡", layout="wide")

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = ""
if "last_source" not in st.session_state:
    st.session_state.last_source = ""
if "last_model" not in st.session_state:
    st.session_state.last_model = ""

selected_model = None
active_api_key = ""

with st.sidebar:
    st.header("⚙️ Конфигурация ИИ")
    
    provider_name = st.selectbox("Провайдер ИИ:", list(PROVIDERS.keys()))
    p_info = PROVIDERS[provider_name]
    
    # Настройка Base URL (для кастомного провайдера можно менять прямо в UI)
    if p_info.get("custom_url"):
        active_base_url = st.text_input("Base URL (OpenAI-совместимый):", value=p_info["base_url"]).strip()
    else:
        active_base_url = p_info["base_url"]

    # Безопасная работа с ключами для ЛЮБОГО провайдера
    env_key_value = clean_token(os.getenv(p_info["env_key"], "")) if p_info["env_key"] else ""

    if p_info["needs_key"]:
        if env_key_value:
            # Если ключ уже есть в .env — не показываем никаких плашек и полей,
            # даём только аккуратный переключатель "Свой API-ключ"
            use_custom_key = st.toggle("🔑 Свой API-ключ", value=False)
            if use_custom_key:
                active_api_key = st.text_input(
                    "Введите API Key:",
                    type="password",
                    value="",
                    placeholder=p_info["placeholder"]
                )
            else:
                active_api_key = env_key_value
        else:
            # Если в .env ключа для этого провайдера нет — сразу просим ввести
            active_api_key = st.text_input(
                "Введите API Key:",
                type="password",
                value="",
                placeholder=p_info["placeholder"],
                help=f"Или добавьте {p_info['env_key']} в файл .env, чтобы не вводить его вручную."
            )
    else:
        active_api_key = "ollama"

    st.divider()

    # Загрузка и выбор модели
    if st.button("🔄 Обновить список моделей", use_container_width=True):
        _cached_fetch_models.clear()

    if not p_info["needs_key"] or active_api_key:
        with st.spinner("Загрузка доступных моделей..."):
            models_list, err_msg = get_provider_models(
                active_base_url, active_api_key, p_info["needs_key"]
            )
        
        if models_list:
            # Позволяем выбрать из списка или переключиться на ручной ввод названия модели
            manual_model_mode = st.toggle("✏️ Ввести ID модели вручную", value=False)
            if manual_model_mode:
                selected_model = st.text_input("ID модели:", value=p_info["default_model"]).strip()
            else:
                selected_model = st.selectbox(
                    f"Модель ({len(models_list)} доступно):",
                    options=models_list,
                    index=0
                )
        else:
            selected_model = st.text_input(
                "ID модели (список недоступен, введите вручную):", 
                value=p_info["default_model"]
            ).strip()
            if err_msg and not p_info["needs_key"]:
                st.caption("⚠️ Сервер Ollama не ответил на `localhost:11434`.")
            elif err_msg:
                st.caption(f"ℹ️ Автозагрузка списка не поддерживается эндпоинтом, укажите модель вручную.")
    else:
        st.info("🔑 Введите API-ключ выше для выбора модели.")

    st.divider()
    max_chars = st.slider(
        "Лимит символов текста:",
        min_value=5_000,
        max_value=150_000,
        value=40_000,
        step=5_000
    )

st.title("⚡ Omni-Extractor AI")
st.caption("Анализ и выжимка из YouTube, веб-статей и PDF/TXT документов с поддержкой любого ИИ")

col1, col2 = st.columns([1, 1])

extracted_text = ""
source_title = ""

with col1:
    st.subheader("1. Источник данных")
    src_type = st.radio("Формат:", ["🔗 Ссылка (YouTube / Web)", "📄 Файл (PDF / TXT / MD)"], horizontal=True)
    
    if "Ссылка" in src_type:
        url_input = st.text_input("Вставьте URL:", placeholder="https://www.youtube.com/watch?v=... или адрес статьи")
        if url_input:
            url_input = url_input.strip()
            try:
                if "youtube.com" in url_input or "youtu.be" in url_input:
                    with st.spinner("Извлечение субтитров YouTube..."):
                        extracted_text = fetch_youtube_transcript(url_input)
                        source_title = f"YouTube ({extract_youtube_id(url_input)})"
                else:
                    with st.spinner("Загрузка и очистка веб-страницы..."):
                        extracted_text = fetch_web_article(url_input)
                        source_title = f"Web: {url_input[:50]}"
            except Exception as e:
                st.error(f"Ошибка загрузки источника: {e}")
    else:
        uploaded_file = st.file_uploader("Загрузите документ", type=["pdf", "txt", "md"])
        if uploaded_file:
            try:
                with st.spinner("Чтение файла..."):
                    if uploaded_file.name.lower().endswith(".pdf"):
                        extracted_text = fetch_pdf_text(uploaded_file.getvalue())
                    else:
                        extracted_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                    source_title = uploaded_file.name
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {e}")

    if extracted_text:
        total_len = len(extracted_text)
        if total_len > max_chars:
            st.warning(f"Загружено **{total_len:,}** симв., но текст будет обрезан до **{max_chars:,}** симв. согласно настройкам.")
        else:
            st.success(f"Загружено: **{source_title}** ({total_len:,} символов)")
        
        with st.expander("👁️ Предпросмотр извлеченного текста"):
            st.text_area("Исходный текст:", value=extracted_text[:5000] + ("..." if total_len > 5000 else ""), height=200, disabled=True)

with col2:
    st.subheader("2. Параметры задачи")
    task_preset = st.selectbox(
        "Что сделать:",
        [
            "Главная мысль + Детальный конспект + Выводы",
            "Deep-Dive FAQ (Скрытые нюансы и разбор кейсов)",
            "Тест для проверки знаний (10 вопросов с ответами)",
            "Тайм-менеджмент: пошаговый план действий (Action Items)",
            "Свой запрос"
        ]
    )
    
    if task_preset == "Свой запрос":
        custom_task = st.text_area("Ваш запрос:", placeholder="Например: Выпиши все упомянутые книги, инструменты и ссылки...")
    else:
        custom_task = task_preset

    has_valid_auth = (not p_info["needs_key"]) or bool(clean_token(active_api_key))
    can_run = bool(extracted_text) and bool(custom_task.strip()) and bool(selected_model) and has_valid_auth
    
    start_btn = st.button(
        f"🚀 Запустить анализ ({selected_model or 'выберите модель'})", 
        type="primary", 
        disabled=not can_run,
        use_container_width=True
    )

st.divider()

if start_btn and can_run:
    trimmed_text = extracted_text[:max_chars]
    
    system_prompt = (
        "Ты — профессиональный аналитик данных и редактор. "
        "Анализируй предоставленный текст строго по фактам, без выдумывания информации. "
        "Дай структурированный, глубокий и легко читаемый ответ в формате Markdown на русском языке."
    )
    user_prompt = f"Источник: {source_title}\n\nКОНТЕНТ:\n{trimmed_text}\n\nЗАДАЧА:\n{custom_task}"
    
    st.markdown(f"#### 🧠 Результат анализа | Провайдер: `{provider_name}` | Модель: `{selected_model}`")
    
    try:
        stream = stream_llm(
            prompt=user_prompt,
            system=system_prompt,
            base_url=active_base_url,
            api_key=clean_token(active_api_key),
            model_name=selected_model
        )
            
        full_response = st.write_stream(stream)
        st.session_state.analysis_result = full_response
        st.session_state.last_source = source_title
        st.session_state.last_model = f"{provider_name} — {selected_model}"

    except Exception as e:
        st.error(
            f"**Ошибка при генерации (`{selected_model}`):** {e}\n\n"
            "💡 *Совет: если эта модель перегружена или недоступна, выберите другую модель в списке слева или переключите провайдера.*"
        )

elif st.session_state.analysis_result:
    st.markdown(f"#### 🧠 Результат анализа ({st.session_state.last_source} | `{st.session_state.last_model}`)")
    st.markdown(st.session_state.analysis_result)

if st.session_state.analysis_result:
    st.download_button(
        label="📥 Скачать результат (.md)",
        data=st.session_state.analysis_result,
        file_name="omni_summary.md",
        mime="text/markdown"
    )