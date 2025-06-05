# для настройки и управления логированием в приложении.
# https://docs.python.org/3/library/logging.html
import logging

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Создает и настраивает логгер с указанным именем и уровнем логирования.
    
    Настраивает вывод логов в консоль и опционально в файл с единым форматом:
    `время - имя_логгера - уровень - сообщение`
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
