# 2 класса для разделения текста на части с контролем длины и перекрытия.
# далеко не оптимальные, но TextSplitter можно использовать даже так. Стабильно и просто.

import re

class MarkdownSplitter:
    """
    Класс для разделения текста на части по заголовкам и контролю длины частей. Относительно сложный
    """
    def __init__(self, chunk_size: int = 1000, overlap: int = 200) :
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Паттерны через ГПТ генерируйте, учить не обязательно.
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)
        
    def split_by_headers(self, text: str) -> list:
        """
        Разделяет текст на части по заголовкам. Убирая строки, состоящие из символов #.
        """
        parts = re.split(self.header_pattern, text)
        return [p.strip() for p in parts if p.strip() and not p.startswith('#')]
    
    def char_control_split(self, parts: list[str]) -> list[str]:
        """
        Разделяет части или объединяет их, чтобы длина была от +- chunk_size до chunk_size + overlap.
        """
        result = []
        current_chunk = ""
        
        for part in parts:
            if len(current_chunk) + len(part) - self.overlap > self.chunk_size:
                if current_chunk:
                    result.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += " " + part

        if current_chunk:
            result.append(current_chunk.strip())

        return result
    
    def split_text(self, text: str) -> list[str]:
        """
        Основной метод для разделения текста на части.
        """
        headers = self.split_by_headers(text)
        return self.char_control_split(headers)
    
    
class TextSplitter:
    """
    Класс для разделения текста на части с контролем длины и перекрытия посимвольно.
    Более простой, чем MarkdownSplitter.
    """
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> list[str]:
        """
        Разделяет текст на части с учетом размера и перекрытия.
        """
        chunks = []
        current_chunk = ""

        for char in text:
            current_chunk += char
            if len(current_chunk) >= self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = current_chunk[-self.overlap:]

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

if __name__ == "__main__":
    with open("data/rag.md", "r", encoding="utf-8") as file:
        text = file.read()
    splitter = TextSplitter()
    # splitter = MarkdownSplitter()
    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} ({len(chunk)} characters):")
        print(chunk)
        print("\n" + "="*40 + "\n")