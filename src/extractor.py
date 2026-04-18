import os
import json
import csv
import io
import warnings
from dataclasses import dataclass
from typing import List, Any, Iterator, Optional, Set
import itertools
import zipfile
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - зависит от окружения
    cv2 = None  # type: ignore

try:
    import easyocr  # type: ignore
except ImportError:  # pragma: no cover - зависит от окружения
    easyocr = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - зависит от окружения
    pd = None  # type: ignore

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - зависит от окружения
    fitz = None  # type: ignore

try:
    import docx  # type: ignore
except ImportError:  # pragma: no cover - зависит от окружения
    docx = None  # type: ignore

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - зависит от окружения
    BeautifulSoup = None  # type: ignore

try:
    from striprtf.striprtf import rtf_to_text  # type: ignore
except ImportError:  # pragma: no cover - зависит от окружения
    rtf_to_text = None  # type: ignore


@dataclass
class ExtractionResult:
    text: str
    status: str
    error: Optional[str] = None


class FileExtractor:

    _reader: Any = None

    @staticmethod
    def _run_quietly(func: Any) -> Any:
        """Подавляет служебный шум OCR-библиотек в консоли."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*pin_memory.*no accelerator is found.*",
                category=UserWarning,
            )
            sink = io.StringIO()
            with redirect_stdout(sink), redirect_stderr(sink):
                return func()

    @classmethod
    def _get_reader(cls) -> Any:
        if easyocr is None:
            raise RuntimeError("easyocr is not installed")
        if cls._reader is None:
            cls._reader = cls._run_quietly(
                lambda: easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
            )
        return cls._reader

    @staticmethod
    def extract_text(file_path: str) -> ExtractionResult:
        ext: str = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".pdf":
                if fitz is None:
                    return ExtractionResult("", "unsupported", "PyMuPDF is not installed")
                return ExtractionResult(FileExtractor._extract_pdf(file_path), "ok")
            if ext == ".docx":
                return ExtractionResult(FileExtractor._extract_docx(file_path), "ok")
            if ext == ".doc":
                return ExtractionResult("", "unsupported", "Legacy .doc is not supported")
            if ext in [".csv", ".parquet", ".xls", ".xlsx"]:
                if ext == ".parquet" and pd is None:
                    return ExtractionResult("", "unsupported", "pandas is required for .parquet")
                if ext in [".xls", ".xlsx"] and pd is None:
                    return ExtractionResult("", "unsupported", "pandas is required for Excel formats")
                return ExtractionResult(FileExtractor._extract_table(file_path, ext), "ok")
            if ext == ".json":
                return ExtractionResult(FileExtractor._extract_json(file_path), "ok")
            if ext == ".rtf":
                if rtf_to_text is None:
                    return ExtractionResult("", "unsupported", "striprtf is not installed")
                return ExtractionResult(FileExtractor._extract_rtf(file_path), "ok")
            if ext == ".html":
                if BeautifulSoup is None:
                    return ExtractionResult("", "unsupported", "beautifulsoup4 is not installed")
                return ExtractionResult(FileExtractor._extract_html(file_path), "ok")
            if ext in [".png", ".jpg", ".jpeg", ".tif", ".gif"]:
                if cv2 is None or easyocr is None:
                    return ExtractionResult("", "unsupported", "OCR dependencies are not installed")
                return ExtractionResult(FileExtractor._extract_image(file_path), "ok")
            if ext in [".mp4", ".avi", ".mov", ".mkv"]:
                if cv2 is None or easyocr is None:
                    return ExtractionResult("", "unsupported", "OCR dependencies are not installed")
                return ExtractionResult(FileExtractor._extract_video(file_path), "ok")
            if ext in [".txt", ".md"]:
                return ExtractionResult(FileExtractor._extract_plain_text(file_path), "ok")
            return ExtractionResult("", "unsupported", f"Unsupported extension: {ext or '<none>'}")
        except Exception as exc:
            return ExtractionResult("", "error", str(exc))

    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        if fitz is None:
            raise RuntimeError("PyMuPDF is not installed")
        text_blocks: List[str] = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text_blocks.append(page.get_text())
        return " ".join(text_blocks)

    @staticmethod
    def _extract_docx(file_path: str) -> str:
        if docx is not None:
            doc = docx.Document(file_path)
            return " ".join([p.text for p in doc.paragraphs])

        with zipfile.ZipFile(file_path) as archive:
            xml_bytes = archive.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs: List[str] = []
        for paragraph in root.findall(".//w:p", namespace):
            text_parts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
            paragraph_text = "".join(text_parts).strip()
            if paragraph_text:
                paragraphs.append(paragraph_text)
        return " ".join(paragraphs)

    @staticmethod
    def _extract_rtf(file_path: str) -> str:
        if rtf_to_text is None:
            raise RuntimeError("striprtf is not installed")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file_obj:
            return str(rtf_to_text(file_obj.read()))

    @staticmethod
    def _extract_table(file_path: str, ext: str) -> str:
        if ext == ".csv":
            if pd is not None:
                df = pd.read_csv(file_path, on_bad_lines="skip", encoding="utf-8")
                return str(df.to_string(index=False))
            return FileExtractor._extract_csv_fallback(file_path)
        if pd is None:
            raise RuntimeError("pandas is not installed")
        if ext == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_excel(file_path)
        return str(df.to_string(index=False))

    @staticmethod
    def _extract_csv_fallback(file_path: str) -> str:
        encodings = ["utf-8", "utf-8-sig", "cp1251", "latin-1"]
        last_error: Optional[Exception] = None

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding, errors="strict", newline="") as file_obj:
                    sample = file_obj.read(2048)
                    file_obj.seek(0)
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
                    reader = csv.reader(file_obj, dialect)
                    rows = [" ".join(cell.strip() for cell in row if cell.strip()) for row in reader]
                    return " ".join(row for row in rows if row)
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Unable to read CSV file: {last_error}")

    @staticmethod
    def _extract_json(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file_obj:
            data = json.load(file_obj)
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _extract_html(file_path: str) -> str:
        if BeautifulSoup is None:
            raise RuntimeError("beautifulsoup4 is not installed")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file_obj:
            soup = BeautifulSoup(file_obj.read(), "html.parser")
            return soup.get_text(separator=" ", strip=True)

    @staticmethod
    def _extract_image(file_path: str) -> str:
        """Двойной проход EasyOCR с дедупликацией фрагментов."""
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed")
        reader = FileExtractor._get_reader()
        img = cv2.imread(file_path)

        if img is None:
            return ""

        fragments: List[str] = []
        seen: Set[str] = set()
        candidates = [img, cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)]

        for candidate in candidates:
            result = FileExtractor._run_quietly(
                lambda: reader.readtext(candidate, detail=0, paragraph=True)
            )
            for chunk in result:
                normalized = " ".join(chunk.split())
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    fragments.append(normalized)

        return " ".join(fragments)

    @staticmethod
    def _extract_video(file_path: str) -> str:
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed")
        reader = FileExtractor._get_reader()
        cap = cv2.VideoCapture(file_path)
        fps_raw: Any = cap.get(cv2.CAP_PROP_FPS)
        fps_val: int = int(fps_raw) if fps_raw and int(fps_raw) > 0 else 24

        text_blocks: List[str] = []
        max_seconds: int = 10
        counter: Iterator[int] = itertools.count()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx: int = next(counter)
            sec_info = divmod(frame_idx, fps_val)
            current_sec: int = int(sec_info[0])
            remainder: int = int(sec_info[1])

            if remainder == 0:
                if current_sec >= max_seconds:
                    break

                result = FileExtractor._run_quietly(
                    lambda: reader.readtext(frame, detail=0)
                )
                if result:
                    text_blocks.append(" ".join(result))

        cap.release()
        return " ".join(text_blocks)

    @staticmethod
    def _extract_plain_text(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file_obj:
            return file_obj.read()
