"""
Модуль конфигурации приложения с использованием Pydantic BaseSettings.

Этот модуль загружает настройки приложения из .env файла с использованием
Pydantic для валидации типов данных и значений по умолчанию. BaseSettings
автоматически загружает переменные окружения и применяет соответствующие
преобразования типов.
"""
# https://docs.pydantic.dev/latest/concepts/pydantic_settings/
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    """Настройки приложения, загружаемые из файла .env.
    
    Настройки приложения, загружаемые из файла .env.
    
    Attributes:
        CHUNK_SIZE: Размер текстовых фрагментов при разделении документов.
        CHUNK_OVERLAP: Перекрытие между соседними фрагментами (в символах).
        LLM_PROMPT_PATH: Путь к файлу с промптом для языковой модели.
        RAW_DIR: Директория с исходными документами для обработки.
        INDEX_FAISS_DIR: Директория для хранения FAISS индекса.
        INDEX_FAISS_NAME: Имя файла FAISS индекса.
        MODEL_EMBEDDING: Название модели для создания эмбеддингов.
        EMBEDDINGS_MODEL_DIR: Директория для локальной модели эмбеддингов.
        FAISS_TOP_K: Количество результатов при поиске через FAISS.
        PROMPT_DATA: Промпт для создания эмбеддингов данных.
        PROMPT_QUERY: Промпт для создания эмбеддингов запроса.
        BM25S_TOP_K: Количество результатов при поиске через BM25.
        SPLITTER: Тип сплиттера для разделения текста ('markdown' или 'text').
    """
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    LLM_PROMPT_PATH: str
    RAW_DIR: str
    INDEX_FAISS_DIR: str
    INDEX_FAISS_NAME: str
    MODEL_EMBEDDING: str
    EMBEDDINGS_MODEL_DIR: str
    FAISS_TOP_K: int
    PROMPT_DATA: str
    PROMPT_QUERY: str
    BM25S_TOP_K: int
    SPLITTER: str

    class Config:
        env_file = ".env"
        # env_file_encoding = "utf-8"
settings = Settings()

