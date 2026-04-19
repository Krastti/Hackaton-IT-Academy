import os
import json
import cv2
import pandas as pd
import fitz  # PyMuPDF
import docx
import docx2txt
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
from typing import List, Any, Optional
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

# --- Новые импорты для Surya OCR (Актуальные для версии 0.17+) ---
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

print("Загрузка моделей Surya OCR")

# 1. Сначала загружаем "фундаментальную" базовую модель
foundation_predictor = FoundationPredictor()

# 2. Передаем её в распознаватель текста (это исправит TypeError)
recognition_predictor = RecognitionPredictor(foundation_predictor)

# 3. Загружаем детектор строк (ему foundation не нужен)
detection_predictor = DetectionPredictor()

print("Модели Surya OCR успешно загружены.")

class FileExtractor:
    # Оставляем методы для документов без изменений...

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

    # --- ДОКУМЕНТЫ (Без изменений) ---
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

            if df is None or df.empty: return ''

            table_data = {"__is_table_data__": True, "columns": {}}
            for col in df.columns:
                table_data["columns"][str(col)] = df[col].dropna().astype(str).tolist()
            return json.dumps(table_data, ensure_ascii=False)
        except Exception as e:
            return ''

    @staticmethod
    def _extract_json(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return json.dumps(json.load(f), ensure_ascii=False)

    @staticmethod
    def _extract_html(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return BeautifulSoup(f.read(), 'html.parser').get_text(separator=' ', strip=True)

    @staticmethod
    def _extract_plain_text(file_path: str) -> str:
        for enc in ('utf-8-sig', 'utf-8', 'cp1251', 'latin-1'):
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        return ''

    # --- МЕДИА (Обновлено под Surya OCR v0.17+ с тихим режимом) ---
    @staticmethod
    def _extract_image(file_path: str) -> str:
        try:
            img = Image.open(file_path).convert("RGB")

            # Временно перенаправляем весь мусорный вывод Surya в никуда (os.devnull)
            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull), redirect_stderr(fnull):
                    predictions = recognition_predictor([img], det_predictor=detection_predictor)

            text_blocks = []
            if predictions and predictions[0].text_lines:
                for line in predictions[0].text_lines:
                    text_blocks.append(line.text)

            return ' '.join(text_blocks)

        except Exception as e:
            # Ошибки мы всё равно увидим, так как они ловятся вне "глушилки"
            print(f"[!] Ошибка Surya OCR в файле {os.path.basename(file_path)}: {e}")
        return ''

    @staticmethod
    def _extract_video(file_path: str) -> str:
        cap = cv2.VideoCapture(file_path)
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 24.0
        max_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps_val * 120))
        step = int(fps_val * 2.0)
        text_blocks = []

        for frame_idx in range(0, max_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # То же самое: глушим вывод для каждого кадра видео
                with open(os.devnull, 'w') as fnull:
                    with redirect_stdout(fnull), redirect_stderr(fnull):
                        predictions = recognition_predictor([pil_img], det_predictor=detection_predictor)

                if predictions and predictions[0].text_lines:
                    for line in predictions[0].text_lines:
                        text_blocks.append(line.text)
            except Exception:
                pass

        cap.release()
        unique_blocks = list(dict.fromkeys(text_blocks))
        return ' '.join(unique_blocks)

# Добавьте это в конец вашего файла extractor.py

class GenericExtractor:
    """Обертка для соответствия интерфейсу, ожидаемому в app.py"""
    def extract(self, file_path: Path) -> str:
        # Вызываем ваш статический метод
        return FileExtractor.extract_text(str(file_path))

class ExtractorFactory:
    def __init__(self):
        self._extractor = GenericExtractor()

    def get(self, file_format: str) -> GenericExtractor:
        # Так как FileExtractor.extract_text сам обрабатывает расширения,
        # мы просто возвращаем один и тот же объект
        return self._extractor