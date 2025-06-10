# Оллама - самый простой способ использовать LLM в локальном режиме.
# в данном примере самый "прямой" способ обращения к модели через API /api/chat
# https://ollama.com/download
from configs.config import settings
import requests
import json
from configs.logger_config import setup_logger

logger = setup_logger(__name__)

MODEL_NAME = "qwen2.5:32b"  # или 'llama3', 'mistral' и т.д.
OLLAMA_BASE_URL = "http://localhost:11434"

with open(settings.LLM_PROMPT_PATH) as f:
    llm_prompt = f.read()


def ask_llm(prompt: str, bm25_results: str, system_prompt: str = llm_prompt):
    """
    Обращение к модели через /api/chat (родной API Ollama)
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": MODEL_NAME,  # Укажите вашу модель
        "messages": [
            {
                "role": "system",
                "content": f"{system_prompt}\n\nПОИСКОВЫЙ КОНТЕКСТ:\n{bm25_results}\n\n",
            },
            {"role": "user", "content": prompt},
        ],
        "options": {  # Дополнительные параметры генерации
            "temperature": 0.2,
            "max_tokens": 200,
        },
        "stream": False,  # Если True, то будет возвращаться поток данных
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url, json=payload
        )  # data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Проверить на HTTP ошибки (4xx или 5xx)

        response_data = response.json()
        return response_data["message"]["content"]

    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка HTTP запроса: {e}")
        logger.info(
            "Убедитесь, что Ollama запущена, модель скачана и доступна по указанному URL."
        )
    except Exception as e:
        logger.error(f"Произошла непредвиденная ошибка: {e}")
    return None
