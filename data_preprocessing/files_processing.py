
# Модуль для обработки текстовых файлов и их разделения на чанки.
# https://www.w3schools.com/python/module_os.asp

import os
from utils.splitters import MarkdownSplitter, TextSplitter
from configs.logger_config import setup_logger
from configs.config import settings

logger = setup_logger(__name__)

def process_files_in_directory(directory: str, splitter: str = 'markdown') -> list[str]:
    """
    Обрабатывает файлы в указанной директории, разделяя их на части с помощью заданного сплиттера.
    
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
        chunks = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            split_text = splitter_instance.split_text(text)
            chunks.extend(split_text)
        logger.info(f"Файлы успешно обработаны в директории {directory} с использованием сплиттера {splitter}.")
        logger.info(f"Всего обработано {len(chunks)} чанков.")
    except Exception as e:
        logger.error(f"Ошибка при обработке файлов в директории {directory}: {e}\n\nКажись файлов нет в папке.")
    return chunks
