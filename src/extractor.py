import os
import json
import cv2
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import docx
import docx2txt
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
from typing import List, Any, Optional
from rapidocr_onnxruntime import RapidOCR


class FileExtractor:
    _ocr: Optional[RapidOCR] = None

    @classmethod
    def _get_ocr(cls) -> RapidOCR:
        if cls._ocr is None:
            # RapidOCR автоматически определяет угол наклона текста (use_angle_cls=True при вызове)
            cls._ocr = RapidOCR()
        return cls._ocr

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

    # --- ДОКУМЕНТЫ ---
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
        parts: List[str] = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        return ' '.join(parts)

    @staticmethod
    def _extract_doc(file_path: str) -> str:
        try:
            text = docx2txt.process(file_path)
            if text: return text
        except Exception:
            pass

        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            text_utf16 = content.decode('utf-16-le', errors='ignore')
            text_cp1251 = content.decode('cp1251', errors='ignore')
            import re
            cleaned_utf16 = re.sub(r'[^\w\s@.,-]', ' ', text_utf16)
            cleaned_cp1251 = re.sub(r'[^\w\s@.,-]', ' ', text_cp1251)
            return cleaned_utf16 + " " + cleaned_cp1251
        except Exception:
            return ''

    @staticmethod
    def _extract_rtf(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return str(rtf_to_text(f.read()))

    # --- СТРУКТУРИРОВАННЫЕ ДАННЫЕ (Таблицы в JSON) ---
    @staticmethod
    def _extract_table(file_path: str, ext: str) -> str:
        ENCODINGS = ['utf-8-sig', 'utf-8', 'cp1251', 'latin-1']
        df = None
        try:
            if ext == '.csv':
                for enc in ENCODINGS:
                    try:
                        df = pd.read_csv(file_path, on_bad_lines='skip', encoding=enc, low_memory=False)
                        break
                    except Exception:
                        continue
            elif ext == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_excel(file_path)

            if df is None or df.empty:
                return ''

            # Упаковываем таблицу в JSON-маркер для быстрого парсинга заголовков в analyzer.py
            table_data = {
                "__is_table_data__": True,
                "columns": {}
            }
            for col in df.columns:
                # Конвертируем в список строк, исключая NaN
                table_data["columns"][str(col)] = df[col].dropna().astype(str).tolist()

            return json.dumps(table_data, ensure_ascii=False)
        except Exception as e:
            print(f"[!] Ошибка чтения таблицы {os.path.basename(file_path)}: {e}")
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

    # --- МЕДИА (RapidOCR) ---
    @staticmethod
    def _extract_image(file_path: str) -> str:
        ocr = FileExtractor._get_ocr()
        try:
            # use_angle_cls=True сам перевернет картинку, если текст боком
            result, _ = ocr(file_path, use_angle_cls=True)
            if result:
                # result: [ [[box], "text", confidence], ... ]
                return ' '.join([item[1] for item in result])
        except Exception as e:
            print(f"[!] Ошибка OCR в файле {os.path.basename(file_path)}: {e}")
        return ''

    @staticmethod
    def _extract_video(file_path: str) -> str:
        ocr = FileExtractor._get_ocr()
        cap = cv2.VideoCapture(file_path)
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 24.0
        max_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps_val * 120))  # Макс 2 мин

        step = int(fps_val * 2.0)  # 1 кадр каждые 2 секунды
        text_blocks = []

        for frame_idx in range(0, max_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break

            try:
                result, _ = ocr(frame, use_angle_cls=True)
                if result:
                    text_blocks.append(' '.join([item[1] for item in result]))
            except Exception:
                pass

        cap.release()
        return ' '.join(text_blocks)