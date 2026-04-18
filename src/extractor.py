import os
import json
import cv2  # type: ignore
import easyocr  # type: ignore
import pandas as pd  # type: ignore
import fitz  # PyMuPDF  # type: ignore
import docx  # type: ignore
import docx2txt  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
from striprtf.striprtf import rtf_to_text  # type: ignore
from typing import List, Any, Dict, Optional, Iterator
import itertools


class FileExtractor:

    _reader: Optional[easyocr.Reader] = None

    @classmethod
    def _get_reader(cls) -> easyocr.Reader:
        if cls._reader is None:
            cls._reader = easyocr.Reader(['ru', 'en'], gpu=False)
        return cls._reader

    @staticmethod
    def extract_text(file_path: str) -> str:
        ext: str = os.path.splitext(file_path)[1].lower()

        try:
            if ext == '.pdf':
                return FileExtractor._extract_pdf(file_path)
            elif ext == '.docx':
                return FileExtractor._extract_docx(file_path)
            elif ext == '.doc':
                return FileExtractor._extract_doc(file_path)
            elif ext in ('.csv', '.parquet', '.xls', '.xlsx'):
                return FileExtractor._extract_table(file_path, ext)
            elif ext == '.json':
                return FileExtractor._extract_json(file_path)
            elif ext == '.rtf':
                return FileExtractor._extract_rtf(file_path)
            elif ext == '.html':
                return FileExtractor._extract_html(file_path)
            elif ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif'):
                return FileExtractor._extract_image(file_path)
            elif ext in ('.mp4', '.avi', '.mov', '.mkv'):
                return FileExtractor._extract_video(file_path)
            elif ext in ('.txt', '.md', '.log'):
                return FileExtractor._extract_plain_text(file_path)
            else:
                return ''
        except Exception as e:
            print(f'[!] Ошибка при обработке {os.path.basename(file_path)}: {e}')
            return ''

    # ------------------------------------------------------------------
    # Документы
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        text_blocks: List[str] = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text_blocks.append(page.get_text())
        return ' '.join(text_blocks)

    @staticmethod
    def _extract_docx(file_path: str) -> str:
        doc = docx.Document(file_path)
        # Параграфы + ячейки таблиц
        parts: List[str] = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        return ' '.join(parts)

    @staticmethod
    def _extract_doc(file_path: str) -> str:
        """Извлечение текста из .doc через docx2txt (не требует LibreOffice)."""
        try:
            return docx2txt.process(file_path) or ''
        except Exception as e:
            print(f'[!] docx2txt не смог открыть {os.path.basename(file_path)}: {e}')
            return ''

    @staticmethod
    def _extract_rtf(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return str(rtf_to_text(f.read()))

    # ------------------------------------------------------------------
    # Структурированные данные
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_table(file_path: str, ext: str) -> str:
        """Чтение таблиц с последовательным перебором кодировок для CSV."""
        ENCODINGS: List[str] = ['utf-8-sig', 'utf-8', 'cp1251', 'latin-1']
        try:
            if ext == '.csv':
                df: Optional[Any] = None
                for enc in ENCODINGS:
                    try:
                        df = pd.read_csv(
                            file_path,
                            on_bad_lines='skip',
                            encoding=enc,
                            low_memory=False,
                        )
                        break  # успешно прочитали
                    except (UnicodeDecodeError, Exception):
                        continue
                if df is None:
                    return ''
            elif ext == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_excel(file_path)
            return str(df.to_string(index=False))
        except Exception:
            return ''

    @staticmethod
    def _extract_json(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _extract_html(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text(separator=' ', strip=True)

    @staticmethod
    def _extract_plain_text(file_path: str) -> str:
        for enc in ('utf-8-sig', 'utf-8', 'cp1251', 'latin-1'):
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        return ''

    # ------------------------------------------------------------------
    # Медиафайлы (OCR)
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_for_ocr(img: Any) -> Any:
        """ЧБ + адаптивный порог для улучшения распознавания документов.

        Адаптивный порог лучше equalizeHist для неравномерного освещения
        (документы с тенями, блики, тёмные края).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31, C=10
        )
        return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _extract_image(file_path: str) -> str:
        """Двойной проход EasyOCR: горизонталь + поворот на 90°.

        Каждый кадр предобрабатывается в ЧБ с адаптивным порогом
        для лучшего распознавания текста на документах.
        """
        reader = FileExtractor._get_reader()
        img = cv2.imread(file_path)

        if img is None:
            return ''

        img_proc = FileExtractor._preprocess_for_ocr(img)
        res_h: List[str] = reader.readtext(img_proc, detail=0, paragraph=True)
        text_horizontal: str = ' '.join(res_h)

        img_rotated = cv2.rotate(img_proc, cv2.ROTATE_90_COUNTERCLOCKWISE)
        res_v: List[str] = reader.readtext(img_rotated, detail=0, paragraph=True)
        text_vertical: str = ' '.join(res_v)

        return text_horizontal + ' ' + text_vertical

    @staticmethod
    def _extract_video(file_path: str) -> str:
        """OCR ключевых кадров: с логированием сырого текста."""
        reader = FileExtractor._get_reader()
        cap = cv2.VideoCapture(file_path)
        fps_raw: Any = cap.get(cv2.CAP_PROP_FPS)
        fps_val: float = float(fps_raw) if fps_raw and fps_raw > 0 else 24.0

        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds: float = total_frames / fps_val
        max_seconds: float = min(total_seconds, 120.0)

        # Шаг: 1 кадр каждые 2 секунды (ускорит дебаг и уменьшит дубли)
        step: float = fps_val * 2.0

        text_blocks: List[str] = []
        frame_idx: int = 0

        # --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
        log_file_path = f"{file_path}_ocr_log.txt"
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"=== Лог OCR для видео: {file_path} ===\n")
        print(f"\n[*] Дебаг: логи OCR для видео пишутся в файл {log_file_path}")
        # -----------------------------

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            current_sec: float = frame_idx / fps_val
            if current_sec > max_seconds:
                break

            frame_processed = FileExtractor._preprocess_for_ocr(frame)
            frame_texts = []  # Собираем текст текущего кадра для лога

            # Горизонтальное чтение
            result_h: List[str] = reader.readtext(frame_processed, detail=0, paragraph=True)
            if result_h:
                text_h = ' '.join(result_h)
                text_blocks.append(text_h)
                frame_texts.append(f"[Гориз]: {text_h}")

            # Вертикальное чтение
            frame_rotated = cv2.rotate(frame_processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
            result_v: List[str] = reader.readtext(frame_rotated, detail=0, paragraph=True)
            if result_v:
                text_v = ' '.join(result_v)
                text_blocks.append(text_v)
                frame_texts.append(f"[Верт]: {text_v}")

            # --- ЗАПИСЬ В ЛОГ И ВЫВОД В КОНСОЛЬ ---
            if frame_texts:
                log_message = f"[{current_sec:.1f} сек] " + " | ".join(frame_texts)
                print(log_message)  # Вывод в терминал
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(log_message + '\n')
            # --------------------------------------

            frame_idx = int(frame_idx + step)

        cap.release()
        return ' '.join(text_blocks)