"""
Основной модуль RAG-системы для ответов на вопросы по документам.

Этот модуль объединяет все компоненты RAG-системы:
1. Обработку текстовых файлов и их разделение на чанки
2. Загрузку модели для создания эмбеддингов
3. Создание векторного индекса FAISS для семантического поиска
4. Применение алгоритма BM25 для уточнения результатов
5. Генерацию ответов с использованием LLM

Схема работы:
- Загрузка документов → Разделение на чанки → Создание эмбеддингов → Индексация
- При получении запроса: FAISS поиск → BM25 ранжирование → Генерация ответа
"""
from configs.logger_config import setup_logger
from data_preprocessing.files_processing import process_files_in_directory
from utils.load_model_hf import load_model
from utils.llm_client import ask_llm
from search.faiss_index import VectorStoreFAISS
from search.bm25_rerank import BM25S
from configs.config import settings

logger = setup_logger(__name__)

def main():
    # ЗАГРУЗКА МОДЕЛИ ЭМБЕДДИНГОВ
    model = load_model()

    # ЧТЕНИЕ ФАЙЛОВ ИЗ ПАПКИ DATA
    #Сплитер из файла data_preprocessing/files_processing.py
    files_chunks = process_files_in_directory(settings.RAW_DIR, splitter=settings.SPLITTER) 

    # СОЗДАНИЕ FAISS ИНДЕКСА ВЕКТОРНОЕ ХРАНИЛИЩЕ
    # Из файла search/faiss_index.py
    faiss_dm = VectorStoreFAISS(
        chunks=files_chunks, 
        model=model,
        prompt_data=settings.PROMPT_DATA,
        prompt_query=settings.PROMPT_QUERY
    )

    # Основной цикл запросов
    while True:
        query = input("Введите ваш запрос: ")
        if query.lower() in ["exit", "quit"]:
            print("Выход из программы.")
            break
        
        # ПЕРВАЯ ЧАСТЬ ПОИСКА - ПОЛУЧЕНИЕ ЧАНКОВ ДЛЯ ОТВЕТА ПО ВЕКТОРНОЙ БЛИЗОСТИ
        chunks_from_faiss = faiss_dm.faiss_search(query=query)
        logger.info(f"Получены чанки({len(chunks_from_faiss)}) из FAISS для запроса: {query}")

        # ВТОРАЯ ЧАСТЬ ПОИСКА - РЕРАНКИНГ С ПОМОЩЬЮ BM25
        bm25 = BM25S(chunks=chunks_from_faiss, top_k=settings.BM25S_TOP_K)
        bm25_results = bm25.get_chunks_for_answering(query)
        logger.info(f"Финальные чанки Готовы")
        full_answer = ask_llm(prompt=query, bm25_results=bm25_results)
        print(f"Ответ:\n\n{full_answer}")


if __name__ == "__main__":
    main()
