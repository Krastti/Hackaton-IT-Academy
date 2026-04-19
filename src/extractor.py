import logging
import json
import re
import pdfplumber
import pandas as pd
import easyocr
import numpy as np
import cv2
import docx
import docx2txt

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, file_path: Path) -> str:
        """Извлекает текст из файла и возвращает строку"""

    def _read_text(self, file_path: Path) -> str:
        """Читает plain-text файл с перебором кодировок."""
        for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
            try:
                return file_path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return ""

class _EasyOCRReader:
    _instance: Optional[easyocr.Reader] = None

    @classmethod
    def get(cls) -> easyocr.Reader:
        if cls._instance is None:
            logger.info("Инициализация EasyOCR reader...")
            cls._instance = easyocr.Reader(["ru", "en"], gpu=False)
        return cls._instance

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
        doc = docx.Document(file_path)
        # Параграфы + ячейки таблиц
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        parts.append(cell.text)
        result = "\n".join(parts)
        logger.info("DOCX извлечён: %s (%d символов)", file_path.name, len(result))
        return result


class DocExtractor(BaseExtractor):
    """Старый формат .doc — пробуем docx2txt, fallback на бинарное чтение."""

    def extract(self, file_path: Path) -> str:
        try:
            text = docx2txt.process(str(file_path))
            if text and text.strip():
                logger.info("DOC извлечён через docx2txt: %s", file_path.name)
                return text
        except Exception:
            pass

        # Fallback: бинарное чтение с поиском кириллицы и цифр
        try:
            content = file_path.read_bytes()
            text_utf16 = content.decode("utf-16-le", errors="ignore")
            text_cp1251 = content.decode("cp1251", errors="ignore")
            cleaned = re.sub(r"[^\w\s@.,\-]", " ", text_utf16 + " " + text_cp1251)
            logger.info("DOC извлечён через бинарное чтение: %s", file_path.name)
            return cleaned
        except Exception as e:
            logger.error("Ошибка чтения DOC %s: %s", file_path.name, e)
            return ""


class RTFExtractor(BaseExtractor):
    def extract(self, file_path: Path) -> str:
        from striprtf.striprtf import rtf_to_text
        result = rtf_to_text(self._read_text(file_path))
        logger.info("RTF извлечён: %s (%d символов)", file_path.name, len(result))
        return result


class HTMLExtractor(BaseExtractor):
    def extract(self, file_path: Path) -> str:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(self._read_text(file_path), "html.parser")
        result = soup.get_text(separator=" ", strip=True)
        logger.info("HTML извлечён: %s (%d символов)", file_path.name, len(result))
        return result


class PlainTextExtractor(BaseExtractor):
    def extract(self, file_path: Path) -> str:
        result = self._read_text(file_path)
        logger.info("Текст извлечён: %s (%d символов)", file_path.name, len(result))
        return result

class CSVExtractor(BaseExtractor):
    def extract(self, file_path: Path) -> str:
        df = None
        for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
            try:
                df = pd.read_csv(
                    file_path, dtype=str, encoding=enc,
                    on_bad_lines="skip", low_memory=False,
                )
                break
            except (UnicodeDecodeError, Exception):
                continue
        if df is None:
            return ""
        # to_csv быстрее to_string — не вычисляет ширину колонок
        result = df.fillna("").to_csv(index=False, sep=" ", na_rep="")
        logger.info("CSV извлечён: %s (%d строк)", file_path.name, len(df))
        return result

class ExcelExtractor(BaseExtractor):
    def extract(self, file_path: Path) -> str:
        df = pd.read_excel(file_path, dtype=str).fillna("")
        result = df.to_csv(index=False, sep=" ", na_rep="")
        logger.info("Excel извлечён: %s (%d строк)", file_path.name, len(df))
        return result

class ParquetExtractor(BaseExtractor):
    def extract(self, file_path: Path) -> str:
        df = pd.read_parquet(file_path).astype(str).fillna("")
        result = df.to_csv(index=False, sep=" ", na_rep="")
        logger.info("Parquet извлечён: %s (%d строк)", file_path.name, len(df))
        return result

class JSONExtractor(BaseExtractor):
    def extract(self, file_path: Path) -> str:
        with file_path.open(encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        result = json.dumps(data, ensure_ascii=False)
        logger.info("JSON извлечён: %s (%d символов)", file_path.name, len(result))
        return result

class ImageExtractor(BaseExtractor):
    """
    OCR через EasyOCR.
    Препроцессинг: адаптивный порог для неравномерного освещения.
    Двойной проход: горизонталь + поворот 90° для повёрнутых документов.
    """

    @staticmethod
    def preprocess(img: np.ndarray) -> np.ndarray:
        """Адаптивный порог — лучше справляется с тенями и бликами на документах."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31, C=10,
        )
        return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    def extract(self, file_path: Path) -> str:
        reader = _EasyOCRReader.get()

        # numpy fromfile корректно читает пути с кириллицей на Windows
        img_array = np.fromfile(str(file_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("Не удалось декодировать изображение: %s", file_path.name)
            return ""

        img_proc = self.preprocess(img)

        # Горизонтальный проход
        res_h = reader.readtext(img_proc, detail=0, paragraph=True, workers=0)
        text_h = " ".join(res_h)

        # Вертикальный проход — документ мог быть сфотографирован повёрнутым
        img_rotated = cv2.rotate(img_proc, cv2.ROTATE_90_COUNTERCLOCKWISE)
        res_v = reader.readtext(img_rotated, detail=0, paragraph=True, workers=0)
        text_v = " ".join(res_v)

        result = (text_h + " " + text_v).strip()
        logger.info("Изображение распознано: %s (%d символов)", file_path.name, len(result))
        return result

class VideoExtractor(BaseExtractor):
    """OCR каждые 2 секунды, первые 2 минуты видео."""

    def extract(self, file_path: Path) -> str:
        reader = _EasyOCRReader.get()
        cap = cv2.VideoCapture(str(file_path))

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_seconds = min(total_frames / fps, 120.0)
        step = fps * 2.0

        text_blocks = []
        frame_idx = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame_idx / fps > max_seconds:
                break

            frame_proc = ImageExtractor.preprocess(frame)
            try:
                res = reader.readtext(frame_proc, detail=0, paragraph=True, workers=0)
                if res:
                    text_blocks.append(" ".join(res))
            except Exception as e:
                logger.warning("OCR кадра упал: %s", e)

            frame_idx = int(frame_idx + step)

        cap.release()
        result = " ".join(text_blocks)
        logger.info("Видео обработано: %s (%d символов)", file_path.name, len(result))
        return result

class ExtractorFactory:
    """
    Возвращает нужный экстрактор по расширению файла
    """

    _CLASSES: dict[str, type] = {
        # Документы
        "pdf":     PDFExtractor,
        "docx":    DocxExtractor,
        "doc":     DocExtractor,
        "rtf":     RTFExtractor,
        "html":    HTMLExtractor,
        "txt":     PlainTextExtractor,
        "md":      PlainTextExtractor,
        # Таблицы
        "csv":     CSVExtractor,
        "xls":     ExcelExtractor,
        "xlsx":    ExcelExtractor,
        "parquet": ParquetExtractor,
        # Данные
        "json":    JSONExtractor,
        # Изображения
        "png":     ImageExtractor,
        "jpg":     ImageExtractor,
        "jpeg":    ImageExtractor,
        "gif":     ImageExtractor,
        "bmp":     ImageExtractor,
        "tif":     ImageExtractor,
        "tiff":    ImageExtractor,
        # Видео
        "mp4":     VideoExtractor,
        "avi":     VideoExtractor,
        "mov":     VideoExtractor,
        "mkv":     VideoExtractor,
    }

    def __init__(self):
        self._instances: dict[str, BaseExtractor] = {}

    def get(self, file_format: str) -> BaseExtractor:
        key = str(file_format).lower().lstrip(".")
        cls = self._CLASSES.get(key)
        if cls is None:
            raise ValueError(f"Неподдерживаемый формат: '{file_format}'")
        if key not in self._instances:
            self._instances[key] = cls()
        return self._instances[key]

    def supported_formats(self) -> list[str]:
        return list(self._CLASSES.keys())
