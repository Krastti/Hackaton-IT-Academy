import logging
import pdfplumber
import pandas as pd
import easyocr

from abc import ABC, abstractmethod
from pathlib import Path
from docx import Document

logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, file_path: Path) -> str:
        """Извлекает текст из файла и возвращает строку"""

    def _read_text(self, file_path: Path, encoding: str = "utf-8") -> str:
        """Вспомогательный метод для чтения plain-text файлов"""
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            return file_path.read_text(encoding="cp1251")

class PDFExtractor(BaseExtractor):
    def extract(self, file_path: Path) -> str:
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        result = "\n".join(text_parts)
        logger.info("PDF извлечён: %s (%d символов)", file_path.name, len(result))
        return result

class DocxExtractor(BaseExtractor):
    def extract(self, file_path: Path) -> str:
        doc = Document(file_path)
        result = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        logger.info("DOCX извлечён: %s (%d символов)", file_path.name, len(result))
        return result

class CSVExtractor(BaseExtractor):
    def extract(self, file_path: Path) -> str:
        df = pd.read_csv(file_path, dtype=str).fillna("")
        result = df.to_string(index=False)
        logger.info("CSV извлечён: %s (%d строк)", file_path.name, len(df))
        return result

class ExcelExtractor(BaseExtractor):
    def extract(self, file_path: Path) -> str:
        df = pd.read_excel(file_path, dtype=str).fillna("")
        result = df.to_string(index=False)
        logger.info("Excel извлечён: %s (%d строк)", file_path.name, len(df))
        return result

class JSONExtractor(BaseExtractor):
    def extract(self, file_path: Path) -> str:
        result = self._read_text(file_path)
        logger.info("JSON извлечён: %s (%d символов)", file_path.name, len(result))
        return result

class PlainTextExtractor(BaseExtractor):
    def extract(self, file_path: Path) -> str:
        result = self._read_text(file_path)
        logger.info("Текст извлечён: %s (%d символов)", file_path.name, len(result))
        return result

class ImageExtractor(BaseExtractor):
    """OCR для изображений через pytesseract."""
    def __init__(self):
        self._reader = easyocr.Reader(["en", "ru"], gpu=False)

    def extract(self, file_path: Path) -> str:
        results = self._reader.readtext(str(file_path), detail=0)
        result = "\n".join(results)
        logger.info("Изображение распознано: %s (%d символов)", file_path.name, len(result))
        return result

class ExtractorFactory:
    """
    Возвращает нужный экстрактор по расширению файла
    """

    _REGISTRY: dict[str, BaseExtractor] = {
        # Документы
        "pdf": PDFExtractor(),
        "docx": DocxExtractor(),
        "doc": DocxExtractor(),
        "rtf": PlainTextExtractor(),
        "html": PlainTextExtractor(),
        "txt": PlainTextExtractor(),
        # Таблицы
        "csv": CSVExtractor(),
        "xls": ExcelExtractor(),
        "xlsx": ExcelExtractor(),
        "parquet": CSVExtractor(),
        # Данные
        "json": JSONExtractor(),
        # Изображения
        "png": ImageExtractor(),
        "jpg": ImageExtractor(),
        "jpeg": ImageExtractor(),
        "gif": ImageExtractor(),
        "bmp": ImageExtractor(),
        # Медиа
    }

    def get(self, file_format: str) -> BaseExtractor:
        extractor = self._REGISTRY.get(file_format.lower())
        if extractor is None:
            raise ValueError(f"Неподдерживаемый формат: '{file_format}'")
        return extractor

    def supported_formats(self) -> list[str]:
        return list(self._REGISTRY.keys())