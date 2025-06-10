"""
Модуль гибридного поиска для RAG-системы.

Этот модуль реализует HybridSearch - класс, который объединяет результаты
двух различных поисковых алгоритмов: семантического поиска FAISS и лексического
поиска BM25. Такой подход позволяет получить более качественные результаты
поиска, учитывая как семантическую близость (смысл), так и точное совпадение
ключевых слов.

Основные концепции:
1. Семантический поиск (FAISS) - находит документы по смыслу
2. Лексический поиск (BM25) - находит документы по точным словам
3. Гибридный поиск - комбинирует оба подхода

Методы комбинирования:
- Weighted (взвешенный) - линейная комбинация нормализованных скоров
- RRF (Reciprocal Rank Fusion) - комбинирование на основе рангов документов
"""

import numpy as np
from typing import List, Tuple, Dict
from configs.logger_config import setup_logger

logger = setup_logger(__name__)


class HybridSearch:
    """
    Класс для гибридного поиска, объединяющего FAISS и BM25.

    Гибридный поиск решает проблему ограничений каждого отдельного метода:
    - FAISS отлично понимает семантику, но может пропустить точные термины
    - BM25 точно находит ключевые слова, но не понимает синонимы и контекст

    Объединяя их результаты, мы получаем лучшее из обоих миров.

    Attributes:
        faiss_store: Экземпляр VectorStoreFAISS для семантического поиска
        bm25_engine: Экземпляр BM25S для лексического поиска
        top_k: Количество возвращаемых результатов
        alpha: Вес FAISS в итоговом скоре (для weighted метода)
        method: Метод комбинирования ("weighted" или "rrf")
    """

    def __init__(
        self,
        faiss_store,  # экземпляр VectorStoreFAISS
        bm25_engine,  # экземпляр BM25S
        top_k: int = 10,
        alpha: float = 0.6,  # вес FAISS в итоговом скоре
        method: str = "weighted",  # "weighted" | "rrf"
    ):
        """
        Инициализация гибридного поискового движка.

        Args:
            faiss_store: Экземпляр VectorStoreFAISS для семантического поиска.
                        Содержит векторный индекс и модель эмбеддингов.
            bm25_engine: Экземпляр BM25S для лексического поиска.
                        Содержит токенизированный корпус и BM25 индекс.
            top_k: Максимальное количество документов для возврата.
                  По умолчанию 10 - оптимальное количество для большинства задач.
            alpha: Вес семантического поиска (FAISS) в финальном скоре.
                  - 0.6 означает 60% веса FAISS, 40% веса BM25
                  - Значения ближе к 1 увеличивают важность семантики
                  - Значения ближе к 0 увеличивают важность точных совпадений
            method: Метод комбинирования результатов:
                   - "weighted": взвешенная сумма нормализованных скоров
                   - "rrf": Reciprocal Rank Fusion (основан на рангах)

        Example:
            # Создание гибридного поиска с акцентом на семантику
            hybrid = HybridSearch(faiss_store, bm25_engine, alpha=0.7)

            # Создание с акцентом на точные совпадения
            hybrid = HybridSearch(faiss_store, bm25_engine, alpha=0.3)
        """
        self.faiss_store = faiss_store
        self.bm25_engine = bm25_engine
        self.top_k = top_k
        self.alpha = alpha
        self.method = method

        logger.info(
            f"Инициализирован HybridSearch: method={method}, alpha={alpha}, top_k={top_k}"
        )

    def search(self, query: str) -> List[str]:
        """
        Выполняет гибридный поиск по запросу.

        Этот метод является основной точкой входа для поиска. Он:
        1. Получает результаты от FAISS (семантический поиск)
        2. Получает результаты от BM25 (лексический поиск)
        3. Комбинирует их одним из выбранных методов
        4. Ранжирует результаты и возвращает топ-K документов

        Args:
            query: Поисковый запрос пользователя

        Returns:
            List[str]: Список наиболее релевантных текстовых чанков,
                      отсортированных по убыванию релевантности

        Example:
            results = hybrid.search("Что такое машинное обучение?")
            for chunk in results:
                print(chunk[:100] + "...")
        """
        logger.info(f"Начинаем гибридный поиск для запроса: '{query}'")

        # Шаг 1: Получаем результаты от семантического поиска (FAISS)
        faiss_hits = self._faiss_with_scores(query)
        logger.debug(f"FAISS вернул {len(faiss_hits)} результатов")

        # Шаг 2: Получаем результаты от лексического поиска (BM25)
        bm25_hits = self._bm25_with_scores(query)
        logger.debug(f"BM25 вернул {len(bm25_hits)} результатов")

        # Шаг 3: Комбинируем результаты выбранным методом
        combined = (
            self._combine_weighted(faiss_hits, bm25_hits)
            if self.method == "weighted"
            else self._combine_rrf(faiss_hits, bm25_hits)
        )

        logger.debug(f"Объединено {len(combined)} уникальных документов")

        # Шаг 4: Сортируем по убыванию скора и берем топ-K
        ranked_ids = sorted(combined.items(), key=lambda x: x[1], reverse=True)[
            : self.top_k
        ]
        logger.info(f"Финальные ранжированные ID (документ, скор): {ranked_ids}")

        # Шаг 5: Возвращаем текстовые чанки (а не индексы)
        result_chunks = [self.faiss_store.chunks[idx] for idx, _ in ranked_ids]
        logger.info(f"Гибридный поиск завершен, возвращено {len(result_chunks)} чанков")

        return result_chunks

    # ==================== СЛУЖЕБНЫЕ МЕТОДЫ ====================
    # Эти методы выполняют специфические задачи внутри гибридного поиска

    def _faiss_with_scores(self, query: str) -> List[Tuple[int, float]]:
        """
        Выполняет семантический поиск через FAISS и возвращает результаты со скорами.

        FAISS (Facebook AI Similarity Search) использует векторные эмбеддинги для
        поиска семантически похожих документов. Он понимает смысл запроса,
        а не только точные слова.

        Args:
            query: Поисковый запрос

        Returns:
            List[Tuple[int, float]]: Список кортежей (индекс_документа, скор_близости)
                                   где скор близости от 0 до 1 (чем больше, тем лучше)
        """
        # Кодируем запрос в векторное представление
        query_embedding = self.faiss_store.model.encode(
            [query],
            prompt=self.faiss_store.prompt_query,  # Специальный промпт для запросов
            normalize_embeddings=True,  # Нормализация для косинусного расстояния
        )

        # Ищем ближайшие векторы в индексе FAISS
        distances, indices = self.faiss_store.index.search(query_embedding, self.top_k)

        # ВАЖНО: FAISS возвращает расстояния (чем меньше, тем лучше)
        # Преобразуем их в скоры близости (чем больше, тем лучше)
        similarities = 1 / (1 + distances[0])  # Формула: sim = 1/(1+dist)

        # Возвращаем список (индекс, скор)
        return list(zip(indices[0], similarities))

    def _bm25_with_scores(self, query: str) -> List[Tuple[int, float]]:
        """
        Выполняет лексический поиск через BM25 и возвращает результаты со скорами.

        BM25 (Best Matching 25) - это классический алгоритм поиска, который
        основан на частоте слов в документах и их редкости в коллекции.
        Он отлично находит документы с точными совпадениями ключевых слов.

        Принцип работы BM25:
        1. Разбивает запрос на токены (слова)
        2. Для каждого токена вычисляет его важность (TF-IDF подобная метрика)
        3. Суммирует скоры по всем токенам запроса

        Args:
            query: Поисковый запрос

        Returns:
            List[Tuple[int, float]]: Список кортежей (индекс_документа, BM25_скор)
                                   BM25 скоры могут быть любыми положительными числами
        """
        # Токенизируем запрос (разбиваем на слова, убираем стоп-слова, применяем стемминг)
        query_tokens = self.bm25_engine._tokenize_query(query)

        # Выполняем поиск через BM25 движок
        # retrieve возвращает (массив_индексов, массив_скоров)
        result_indices, scores = self.bm25_engine.retriever.retrieve(
            query_tokens, k=self.top_k
        )

        # Преобразуем в удобный формат: список кортежей (индекс, скор)
        return list(zip(result_indices[0], scores[0]))

    # ==================== МЕТОДЫ КОМБИНИРОВАНИЯ ====================

    def _combine_weighted(
        self,
        faiss_hits: List[Tuple[int, float]],
        bm25_hits: List[Tuple[int, float]],
    ) -> Dict[int, float]:
        """
        Комбинирует результаты FAISS и BM25 методом взвешенного суммирования.

        Этот метод:
        1. Нормализует скоры FAISS и BM25 к диапазону [0, 1]
        2. Вычисляет взвешенную сумму: alpha * faiss_score + (1-alpha) * bm25_score
        3. Возвращает объединенные скоры для всех документов

        Преимущества метода:
        - Простой и понятный
        - Позволяет точно контролировать вклад каждого алгоритма через alpha
        - Работает стабильно на разных типах запросов

        Args:
            faiss_hits: Результаты FAISS в формате [(индекс, скор), ...]
            bm25_hits: Результаты BM25 в формате [(индекс, скор), ...]

        Returns:
            Dict[int, float]: Словарь {индекс_документа: финальный_скор}
                             где финальный_скор = alpha*faiss + (1-alpha)*bm25
        """
        # Нормализуем скоры FAISS к диапазону [0, 1]
        faiss_normalized = self._norm_scores(faiss_hits)
        logger.debug(f"FAISS скоры нормализованы: {len(faiss_normalized)} документов")

        # Нормализуем скоры BM25 к диапазону [0, 1]
        bm25_normalized = self._norm_scores(bm25_hits)
        logger.debug(f"BM25 скоры нормализованы: {len(bm25_normalized)} документов")

        # Находим все уникальные документы из обоих поисков
        all_document_ids = set(faiss_normalized.keys()) | set(bm25_normalized.keys())
        logger.debug(f"Всего уникальных документов: {len(all_document_ids)}")

        # Вычисляем взвешенную комбинацию для каждого документа
        combined_scores = {}
        for doc_id in all_document_ids:
            # Получаем нормализованные скоры (0.0 если документа нет в результатах)
            faiss_score = faiss_normalized.get(doc_id, 0.0)
            bm25_score = bm25_normalized.get(doc_id, 0.0)

            # Вычисляем взвешенную сумму
            combined_scores[doc_id] = (
                self.alpha * faiss_score + (1 - self.alpha) * bm25_score
            )

        logger.debug(f"Комбинирование завершено методом weighted, alpha={self.alpha}")
        return combined_scores

    def _combine_rrf(
        self,
        faiss_hits: List[Tuple[int, float]],
        bm25_hits: List[Tuple[int, float]],
        k: int = 60,
    ) -> Dict[int, float]:
        """
        Комбинирует результаты методом Reciprocal Rank Fusion (RRF).

        RRF - это метод, который фокусируется на позициях (рангах) документов
        в результатах поиска, а не на их абсолютных скорах. Это делает его
        более устойчивым к различиям в масштабах скоров между алгоритмами.

        Принцип работы:
        1. Сортирует результаты каждого алгоритма по убыванию скоров
        2. Для каждого документа вычисляет RRF скор: 1/(k + rank)
        3. Суммирует RRF скоры от разных алгоритмов

        Преимущества RRF:
        - Не зависит от абсолютных значений скоров
        - Дает хорошие результаты без подбора параметров
        - Устойчив к выбросам в скорах

        Args:
            faiss_hits: Результаты FAISS
            bm25_hits: Результаты BM25
            k: Параметр сглаживания RRF (обычно 60)
               Чем больше k, тем меньше влияние позиции

        Returns:
            Dict[int, float]: Словарь {индекс_документа: RRF_скор}
        """

        def calculate_rrf_score(rank: int) -> float:
            """Вычисляет RRF скор для данной позиции (ранга)."""
            return 1.0 / (k + rank)

        combined_scores = {}

        # Обрабатываем результаты FAISS
        # Сортируем по убыванию скоров и присваиваем ранги
        faiss_sorted = sorted(faiss_hits, key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(faiss_sorted):
            # rank начинается с 0, но ранги в RRF с 1
            rrf_score = calculate_rrf_score(rank + 1)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score

        # Обрабатываем результаты BM25
        bm25_sorted = sorted(bm25_hits, key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(bm25_sorted):
            rrf_score = calculate_rrf_score(rank + 1)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score

        logger.debug(f"RRF комбинирование завершено с параметром k={k}")
        logger.debug(f"Обработано FAISS документов: {len(faiss_sorted)}")
        logger.debug(f"Обработано BM25 документов: {len(bm25_sorted)}")

        return combined_scores

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================

    @staticmethod
    def _norm_scores(hits: List[Tuple[int, float]]) -> Dict[int, float]:
        """
        Нормализует скоры к диапазону [0, 1] методом Min-Max нормализации.

        Нормализация необходима потому, что FAISS и BM25 используют разные
        шкалы скоров:
        - FAISS: скоры близости обычно от 0 до 1
        - BM25: скоры могут быть любыми положительными числами

        Min-Max нормализация приводит все скоры к единому диапазону [0, 1]:
        normalized_score = (score - min_score) / (max_score - min_score)

        Это позволяет справедливо комбинировать результаты разных алгоритмов.

        Args:
            hits: Список кортежей (индекс_документа, скор)

        Returns:
            Dict[int, float]: Словарь {индекс_документа: нормализованный_скор}
                             где все скоры в диапазоне [0, 1]

        Example:
            hits = [(1, 0.8), (2, 0.6), (3, 0.9)]
            normalized = _norm_scores(hits)
            # Результат: {1: 0.666, 2: 0.0, 3: 1.0}
            # Скор 0.6 стал 0.0 (минимум), 0.9 стал 1.0 (максимум)
        """
        # Проверяем, есть ли вообще результаты
        if not hits:
            logger.debug("Пустой список результатов для нормализации")
            return {}

        # Извлекаем только скоры из кортежей
        scores = np.array([score for _, score in hits])

        # Находим минимальный и максимальный скоры
        min_score = scores.min()
        max_score = scores.max()

        # Особый случай: все скоры одинаковые
        if max_score == min_score:
            # Если все скоры равны, присваиваем им значение 1.0
            normalized_scores = np.ones_like(scores)
            logger.debug(f"Все скоры одинаковые ({max_score}), присвоено значение 1.0")
        else:
            # Стандартная Min-Max нормализация
            normalized_scores = (scores - min_score) / (max_score - min_score)
            logger.debug(f"Нормализация: min={min_score:.3f}, max={max_score:.3f}")

        # Создаем словарь {индекс_документа: нормализованный_скор}
        result = {
            doc_id: float(norm_score)
            for (doc_id, _), norm_score in zip(hits, normalized_scores)
        }

        logger.debug(f"Нормализовано {len(result)} скоров")
        return result


# ==================== ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ДЛЯ СТУДЕНТОВ ====================

"""
ПРИМЕР 1: Базовое использование гибридного поиска

# Инициализация компонентов
faiss_store = VectorStoreFAISS(chunks, model, prompt_data, prompt_query)
bm25_engine = BM25S(chunks, top_k=10)

# Создание гибридного поиска с акцентом на семантику
hybrid = HybridSearch(
    faiss_store=faiss_store,
    bm25_engine=bm25_engine,
    top_k=5,
    alpha=0.7,  # 70% семантика, 30% точные совпадения
    method="weighted"
)

# Выполнение поиска
results = hybrid.search("Что такое машинное обучение?")
for i, chunk in enumerate(results, 1):
    print(f"{i}. {chunk[:100]}...")

ПРИМЕР 2: Сравнение методов комбинирования

# Weighted подход (подходит, когда нужен контроль баланса)
hybrid_weighted = HybridSearch(faiss_store, bm25_engine, method="weighted", alpha=0.6)

# RRF подход (подходит, когда скоры сильно различаются по масштабу)
hybrid_rrf = HybridSearch(faiss_store, bm25_engine, method="rrf")

query = "deep learning neural networks"
results_weighted = hybrid_weighted.search(query)
results_rrf = hybrid_rrf.search(query)

ПРИМЕР 3: Настройка для разных типов запросов

# Для фактических запросов (больше вес BM25)
hybrid_factual = HybridSearch(faiss_store, bm25_engine, alpha=0.3)

# Для концептуальных запросов (больше вес FAISS)  
hybrid_conceptual = HybridSearch(faiss_store, bm25_engine, alpha=0.8)

# Фактический запрос: "Когда был создан алгоритм backpropagation?"
factual_results = hybrid_factual.search("Когда был создан алгоритм backpropagation?")

# Концептуальный запрос: "Объясни принципы обучения нейронных сетей"
conceptual_results = hybrid_conceptual.search("Объясни принципы обучения нейронных сетей")
"""

# ==================== ТЕХНИЧЕСКИЕ ДЕТАЛИ ДЛЯ ИЗУЧЕНИЯ ====================

"""
АЛГОРИТМИЧЕСКАЯ СЛОЖНОСТЬ:

1. Время выполнения поиска: O(d * log(n) + k)
   где d - размерность векторов, n - количество документов, k - top_k
   
2. Память: O(n * d) для хранения векторного индекса
   
3. Предобработка: O(n * d) для создания FAISS индекса

ПАРАМЕТРЫ НАСТРОЙКИ:

1. alpha (для weighted метода):
   - 0.9-1.0: почти только семантический поиск
   - 0.7-0.8: семантика важнее, но учитываются точные совпадения
   - 0.5-0.6: сбалансированный подход (рекомендуется)
   - 0.2-0.4: точные совпадения важнее семантики
   - 0.0-0.1: почти только лексический поиск

2. k (для RRF метода):
   - 30-40: высокое влияние позиции в результатах
   - 60-80: стандартное значение (рекомендуется)
   - 100+: сниженное влияние позиции

КОГДА ИСПОЛЬЗОВАТЬ КАКОЙ МЕТОД:

Weighted:
✓ Когда вы понимаете характер запросов
✓ Когда нужен точный контроль баланса  
✓ Когда скоры FAISS и BM25 в сопоставимых диапазонах

RRF:
✓ Когда скоры алгоритмов сильно различаются
✓ Когда нужен устойчивый результат без настройки
✓ Когда важны топовые результаты каждого алгоритма
"""
