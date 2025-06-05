# Векторная база FAISS для хранения и поиска по векторной близости
# https://github.com/facebookresearch/faiss/wiki/Getting-started
from configs.config import settings
from configs.logger_config import setup_logger
import faiss
import os

logger = setup_logger(__name__)

class VectorStoreFAISS:
    """
    Класс для создания и управления векторной базой на основе FAISS.
    
    VectorStoreFAISS предоставляет функционал для создания эмбеддингов текстовых фрагментов,
    их индексации в FAISS, а также для выполнения семантического поиска по этим индексам.
    Поддерживает сохранение и загрузку индексов для повторного использования.
    """
    def __init__(self, chunks: list[str], model, top_k: int = None, prompt_data: str = "", prompt_query: str = ""):
        """
        Инициализирует векторную базу FAISS для работы с текстовыми фрагментами.

        Args:
            chunks (list[str]): Список текстовых фрагментов для индексации.
            model: Модель SentenceTransformer для генерации эмбеддингов.
            top_k (int, optional): Количество возвращаемых результатов поиска. 
                По умолчанию берется из настроек.
            prompt_data (str, optional): Подсказка для модели при создании эмбеддингов данных. 
                По умолчанию берется из настроек.
            prompt_query (str, optional): Подсказка для модели при создании эмбеддингов запроса. 
                По умолчанию берется из настроек.
                
        Note:
            При инициализации класса автоматически генерируются эмбеддинги 
            и создается FAISS индекс.
        """
        self.chunks = chunks
        self.model = model
        self.prompt_data = prompt_data if prompt_data else settings.PROMPT_DATA
        self.prompt_query = prompt_query if prompt_query else settings.PROMPT_QUERY
        self.dim = self.model.get_sentence_embedding_dimension()
        self.embeddings = self.generate_embeddings_data()
        self.index = self.create_faiss_index()
        self.save_dir = settings.INDEX_FAISS_DIR
        self.faiss_name = settings.INDEX_FAISS_NAME
        self.top_k = top_k if top_k is not None else settings.FAISS_TOP_K

    def generate_embeddings_data(self) -> list[list[float]]:
        """
        Генерирует эмбеддинги для списка текстовых фрагментов.

        Использует предоставленную модель SentenceTransformer для преобразования
        текстовых фрагментов в числовые векторы (эмбеддинги).

        Returns:
            list[list[float]]: Матрица эмбеддингов, где каждый эмбеддинг представлен
            как список чисел с плавающей запятой.
            
        Raises:
            Exception: При возникновении ошибки в процессе генерации эмбеддингов.
        """
        try:
            embeddings = self.model.encode(self.chunks, prompt=self.prompt_data, show_progress_bar=True, normalize_embeddings=True)
            logger.info(f"Эмбеддинги успешно сгенерированы для {len(self.chunks)} текстов моделью {settings.MODEL_EMBEDDING}.")
            return embeddings
        except Exception as e:
            logger.error(f"Ошибка при генерации эмбеддингов: {e}")
            raise e

    def create_faiss_index(self) -> faiss.Index:
        """
        Создает и наполняет FAISS индекс сгенерированными эмбеддингами.

        Метод инициализирует индекс типа IndexFlatL2 (точный поиск с евклидовой метрикой)
        и добавляет в него сгенерированные эмбеддинги.

        Returns:
            faiss.Index: Созданный и наполненный FAISS индекс.
            
        Raises:
            Exception: При возникновении ошибки в процессе создания индекса.
        """
        try:
            index = faiss.IndexFlatL2(self.dim)
            index.add(self.embeddings)
            logger.info(f"FAISS индекс успешно создан с размерностью {self.dim}.")
            return index
        except Exception as e:
            logger.error(f"Ошибка при создании FAISS индекса: {e}")
            raise e

    def faiss_search(self, query: str) -> list[str]:
        """
        Выполняет семантический поиск по индексу FAISS.

        Преобразует строку запроса в эмбеддинг и находит наиболее
        близкие к нему векторы в индексе FAISS.

        Args:
            query (str): Текстовый запрос для поиска.

        Returns:
            list[str]: Список текстовых фрагментов, наиболее релевантных запросу.
            
        Raises:
            Exception: При возникновении ошибки в процессе поиска.
        """
        try:
            query_embedding = self.model.encode([query], prompt=self.prompt_query, show_progress_bar=True, normalize_embeddings=True)
            distances, indices = self.index.search(query_embedding, self.top_k)
            logger.info(f"FAISS поиск завершен для запроса: {query}")
            best_chunks = [self.chunks[i] for i in indices[0]]
            return best_chunks
        except Exception as e:
            logger.error(f"Ошибка при выполнении FAISS поиска: {e}")
            raise e
        
        
    # НА БУДУЩЕЕ
    def save_index(self) -> None:
        """
        Сохраняет FAISS индекс в файл.
        :return: None
        """
        os.makedirs(self.save_dir, exist_ok=True)
        if not self.index:
            logger.warning("FAISS индекс не создан, сохранение невозможно.")
            return
        try:
            faiss.write_index(self.index, path = self.save_dir  + self.faiss_name)
            logger.info(f"FAISS индекс успешно сохранен в {self.save_dir}.")
        except Exception as e:
            logger.error(f"Ошибка при сохранении FAISS индекса: {e}")
            raise e
        
    def load_index(self) -> faiss.Index:
        """
        Загружает FAISS индекс из файла.

        :return: Загруженный FAISS индекс.
        """
        try:
            index = faiss.read_index(self.save_dir + self.faiss_name)
            logger.info(f"FAISS индекс успешно загружен из {self.save_dir}.")
            return index
        except Exception as e:
            logger.error(f"Ошибка при загрузке FAISS индекса: {e}")
            raise e
