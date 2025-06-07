# Модуль для локальной загрузки модели Hugging Face с использованием SentenceTransformer
# При следующих запусках модель подгружается уже из EMBEDDINGS_MODEL_DIR
import os
from sentence_transformers import SentenceTransformer
from configs.config import settings
from configs.logger_config import setup_logger

logger = setup_logger(__name__)

def load_model() -> SentenceTransformer:
    try:
        os.makedirs(settings.EMBEDDINGS_MODEL_DIR, exist_ok=True)
        
        model = SentenceTransformer(settings.MODEL_EMBEDDING, cache_folder=settings.EMBEDDINGS_MODEL_DIR, trust_remote_code=True)
        logger.info(f"Модель {settings.MODEL_EMBEDDING} успешно загружена и инициализирована в директории {settings.EMBEDDINGS_MODEL_DIR}")
        return model
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели {settings.MODEL_EMBEDDING}: {e}")
        raise e
