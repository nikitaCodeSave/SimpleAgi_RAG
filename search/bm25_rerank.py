# Модуль для реализации поиска с использованием алгоритма BM25.
# https://github.com/xhluca/bm25s
import bm25s
import Stemmer
from configs.config import settings
from configs.logger_config import setup_logger

logger = setup_logger(__name__)


class BM25S:
    """
    Класс для поиска релевантных фрагментов текста с использованием алгоритма BM25.
    """
    def __init__(self, 
                 chunks: list[str],
                 top_k: int = None,
                 method: str = "bm25+",
                 delta: float = 1.0,
                 k1: float = 1.2,
                 b: float = 0.75,
                 stemmer: str = "russian",
                 stopwords: str = "ru") -> str:
        """
        Инициализация поискового движка BM25.
        
        Args:
            chunks: Список текстовых фрагментов для индексации
            top_k: Количество возвращаемых результатов
            method: Метод BM25 ("bm25", "bm25+" и др.)
            delta: Параметр delta для метода bm25+
            k1: Параметр k1 для настройки частоты термина
            b: Параметр b для нормализации длины документа
            stemmer: Язык стемминга ("russian", "english" и т.д.)
            stopwords: Язык стоп-слов ("ru", "en" и т.д.)
        """
        self.chunks = chunks
        self.top_k = top_k if top_k is not None else settings.BM25_TOP_K
        self.stemmer = Stemmer.Stemmer(stemmer)
        self.corpus_tokens = bm25s.tokenize(chunks, stopwords=stopwords, stemmer=self.stemmer)
        self.method = method
        self.delta = delta
        self.k1 = k1
        self.b = b
        self.retriever = bm25s.BM25(method=method,
                                    delta=delta,
                                    k1=k1,
                                    b=b)
        self.index = self.retriever.index(self.corpus_tokens)
        
    def get_chunks_for_answering(self, query: str) -> list[dict]:
        """
        Выполняет поиск по запросу и возвращает топ K результатов.
        
        :Args:
            query: Строка запроса для поиска
            
        Returns:
            Строка с объединенными найденными релевантными фрагментами
        """
        query_tokens = bm25s.tokenize(query, stopwords="ru", stemmer=self.stemmer)
        result, score = self.retriever.retrieve(query_tokens, k=self.top_k)
        results = "\n".join(self.chunks[i] for i in result[0])
        logger.info(f"BM25 поиск завершен. Всего финальных чанков: {len(result[0])}")
        return results

