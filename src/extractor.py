import logging
import pdfplumber
import pandas as pd
import easyocr

from abc import ABC, abstractmethod
from pathlib import Path
from docx import Document
from paddleocr import PaddleOCR

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

class EasyOCRExtractor(BaseExtractor):
    """OCR для изображений через easyocr."""

    def __init__(self):
        self._reader = easyocr.Reader(["en", "ru"], gpu=False)

    def extract(self, file_path: Path) -> str:
        results = self._reader.readtext(str(file_path), detail=0)
        result = "\n".join(results)
        logger.info("EasyOCR распознал: %s (%d символов)", file_path.name, len(result))
        return result

class PaddleOCRExtractor(BaseExtractor):
    """
    OCR через PaddleOCR для паспартов и сканов.
    """

    def __init__(self):
        self._ocr = PaddleOCR(use_angle_cls=True, lang="ru")

    def extract(self, file_path: Path) -> str:
        raw = self._ocr.ocr(str(file_path), cls=True)
        lines = []
        for page in raw:
            if page:
                for line in page:
                    text, confidence = line[1]
                    if confidence > 0.5:
                        lines.append(text)
        result = "\n".join(lines)
        logger.info("PaddleOCR распознал: %s (%d символов)", file_path.name, len(result))
        return result

class ImageExtractor(BaseExtractor):
    """
    Основной экстрактор для изображений.
    Сначала пробует PaddleOCR, при ошибке — fallback на EasyOCR.
    """

    def __init__(self):
        self._paddle = None
        self._easy = EasyOCRExtractor()
        try:
            self._paddle = PaddleOCRExtractor()
            logger.info("ImageExtractor: используется PaddleOCR")
        except Exception as e:
            logger.warning("PaddleOCR недоступен (%s) — используется EasyOCR", e)

    def extract(self, file_path: Path) -> str:
        if self._paddle:
            try:
                return self._paddle.extract(file_path)
            except Exception as e:
                logger.warning("PaddleOCR упал (%s), fallback на EasyOCR", e)
        return self._easy.extract(file_path)

class ExtractorFactory:
    """
    Возвращает нужный экстрактор по расширению файла.
    Экстракторы создаются один раз при первом обращении (ленивая инициализация).
    """

    _CLASSES: dict[str, type] = {
        # Документы
        "pdf":     PDFExtractor,
        "docx":    DocxExtractor,
        "doc":     DocxExtractor,
        "rtf":     PlainTextExtractor,
        "html":    PlainTextExtractor,
        "txt":     PlainTextExtractor,
        # Таблицы
        "csv":     CSVExtractor,
        "xls":     ExcelExtractor,
        "xlsx":    ExcelExtractor,
        "parquet": CSVExtractor,
        # Данные
        "json":    JSONExtractor,
        # Изображения
        "png":     ImageExtractor,
        "jpg":     ImageExtractor,
        "jpeg":    ImageExtractor,
        "gif":     ImageExtractor,
        "bmp":     ImageExtractor,
    }

    def __init__(self):
        # Экземпляры создаются при первом вызове get(), не при загрузке модуля
        self._instances: dict[str, BaseExtractor] = {}

    def get(self, file_format: str) -> BaseExtractor:
        key = file_format.lower()
        cls = self._CLASSES.get(key)
        if cls is None:
            raise ValueError(f"Неподдерживаемый формат: '{file_format}'")
        if key not in self._instances:
            self._instances[key] = cls()
        return self._instances[key]

    def supported_formats(self) -> list[str]:
        return list(self._CLASSES.keys())