
# Модуль для обработки текстовых файлов и их разделения на чанки.
# https://www.w3schools.com/python/module_os.asp

import os
import asyncio
from utils.splitters import MarkdownSplitter, TextSplitter
from configs.logger_config import setup_logger
from configs.config import settings

logger = setup_logger(__name__)

async def read_and_process_file(file_path: str, splitter_instance) -> list[str]:
    """
    Асинхронно читает и обрабатывает один файл.
    
    Args:
        file_path: Путь к файлу.
        splitter_instance: Экземпляр сплиттера для разделения текста.
    
    Returns:
        list[str]: Список чанков из файла.
    """
    def _read_file():
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Используем asyncio.to_thread для чистого asyncio подхода
    text = await asyncio.to_thread(_read_file)
    chunks = await asyncio.to_thread(splitter_instance.split_text, text)
    
    return chunks

async def process_files_in_directory_async(directory: str, splitter: str = 'markdown') -> list[str]:
    """
    Асинхронно обрабатывает файлы в указанной директории, разделяя их на части с помощью заданного сплиттера.
    
    Args:
        directory: Путь к директории с файлами.
        splitter: Тип сплиттера ('markdown' или 'text').
    
    Returns:
        list[str]: Список обработанных текстовых чанков.
        
    Raises:
        FileNotFoundError: Если указанная директория не существует.
        Exception: При возникновении ошибок во время обработки файлов.
    """
    if splitter == 'markdown':
        splitter_instance = MarkdownSplitter(chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
    elif splitter == 'text':
        splitter_instance = TextSplitter(chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
    
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        file_paths = [os.path.join(directory, file) for file in files]
        
        # Обрабатываем все файлы асинхронно
        tasks = [read_and_process_file(file_path, splitter_instance) for file_path in file_paths]
        results = await asyncio.gather(*tasks)
        
        # Объединяем результаты
        chunks = []
        for file_chunks in results:
            chunks.extend(file_chunks)
        
        logger.info(f"Файлы успешно обработаны в директории {directory} с использованием сплиттера {splitter}.")
        logger.info(f"Всего обработано {len(chunks)} чанков.")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке файлов в директории {directory}: {e}\n\nКажись файлов нет в папке.")
        return []
    
    return chunks

def process_files_in_directory(directory: str, splitter: str = 'markdown') -> list[str]:
    """
    Обрабатывает файлы в указанной директории, разделяя их на части с помощью заданного сплиттера.
    Синхронная обертка над асинхронной функцией для обратной совместимости.
    
    Args:
        directory: Путь к директории с файлами.
        splitter: Тип сплиттера ('markdown' или 'text').
    
    Returns:
        list[str]: Список обработанных текстовых чанков.
        
    Raises:
        FileNotFoundError: Если указанная директория не существует.
        Exception: При возникновении ошибок во время обработки файлов.
    """
    return asyncio.run(process_files_in_directory_async(directory, splitter))
