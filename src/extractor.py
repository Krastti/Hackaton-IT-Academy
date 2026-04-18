import json
import importlib
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Sequence

import cv2
import docx
import easyocr
import pandas as pd
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text

logger = logging.getLogger(__name__)

ExtractStatus = Literal['success', 'empty', 'unsupported', 'failed', 'partial']
ImageLike = Any
TableFrame = Any
ExtractorHandler = Callable[['FileExtractor', str, 'ExtractContext'], str]

"""Неизменяемый контейнер параметров, управляющих поведением экстрактора"""
@dataclass(frozen=True) # Данный декоратор делает конфигурацию иммутабельной и потокобезопасной
class ExtractConfig:
    enable_ocr: bool = True # Вкл/Выкл оптическое распознавание. Позволяет откл. Тяжелый OCR на слабых машинах
    image_rotate_passes: int = 3 # Сколько доп поворотов изображения пробовать для OCR. Повышает точность
    video_max_seconds: float = 120.0 # Макс время видео для анализа
    video_fps_sample: float = 1.0 # Частота сэмплирования кадров
    max_pdf_pages: int = 200 # Лимит страниц PDF
    max_table_rows: int = 1000 # Лимит строк для таблиц
    max_json_chars: int = 200000 # Лимит символов в JSON-файлах после парсинга
    max_chars: int = 200000 # Общий лимит итогового текста на файл

'''Стандартизированный объект-результат обработки одного файла.'''
@dataclass
class ExtractResult:
    file_path: str # Полный путь к обработанному файлу
    text: str # Извлеченный и нормализованный текст. Возращает пустую строку, если извлечь не удалось
    status: ExtractStatus # На основе полученного литерала определяет дальнейшую логику обработки
    extractor: str # Имя вызванного хендлера
    warnings: List[str] = field(default_factory=list) # Список предупреждений
    error: Optional[str] = None # Текст исключения

'''
Назначение: мутируемый "контекстный мешок", передаваемый внутрь каждого метода экстрактора.
Содержит текущие настройки и аккумулирует побочные эффекты обработки.
'''
@dataclass
class ExtractContext:
    config: ExtractConfig # Ссылка на активный конфиг. Хендлеры читают из неё лимиты и флаги.
    warnings: List[str] = field(default_factory=list)
    handler_name: str = ''

class FileExtractor:
    _reader: ClassVar[Optional[easyocr.Reader]] = None
    _reader_lock: ClassVar[Lock] = Lock()
    _default_config: ClassVar[ExtractConfig] = ExtractConfig()
    _pdf_module: ClassVar[Optional[Any]] = None
    _pdf_module_lock: ClassVar[Lock] = Lock()
    _pdf_module_checked: ClassVar[bool] = False
    _handlers: ClassVar[Dict[str, str]] = {
        '.pdf': '_extract_pdf',
        '.docx': '_extract_docx',
        '.doc': '_extract_doc',
        '.csv': '_extract_table',
        '.tsv': '_extract_table',
        '.parquet': '_extract_table',
        '.xls': '_extract_table',
        '.xlsx': '_extract_table',
        '.json': '_extract_json',
        '.rtf': '_extract_rtf',
        '.html': '_extract_html',
        '.htm': '_extract_html',
        '.xml': '_extract_html',
        '.png': '_extract_image',
        '.jpg': '_extract_image',
        '.jpeg': '_extract_image',
        '.tif': '_extract_image',
        '.tiff': '_extract_image',
        '.gif': '_extract_image',
        '.bmp': '_extract_image',
        '.webp': '_extract_image',
        '.mp4': '_extract_video',
        '.avi': '_extract_video',
        '.mov': '_extract_video',
        '.mkv': '_extract_video',
        '.txt': '_extract_plain_text',
        '.md': '_extract_plain_text',
        '.log': '_extract_plain_text',
    }

    @classmethod
    def _get_reader(cls) -> easyocr.Reader:
        """
        Возвращает общий экземпляр OCR-ридера.
        :return: Инициализированный easyocr.Reader
        """
        if cls._reader is None:
            with cls._reader_lock:
                if cls._reader is None:
                    cls._reader = easyocr.Reader(['ru', 'en'], gpu=False)
        return cls._reader

    @classmethod
    def _get_pdf_module(cls) -> Optional[Any]:
        """
        Возвращает модуль для работы с PDF, если он доступен в окружении.
        :return: Модуль PDF-обработки или None
        """
        if cls._pdf_module_checked:
            return cls._pdf_module

        with cls._pdf_module_lock:
            if cls._pdf_module_checked:
                return cls._pdf_module

            for module_name in ('pymupdf', 'fitz'):
                try:
                    module = importlib.import_module(module_name)
                except (ModuleNotFoundError, ImportError):
                    continue
                except Exception as e:
                    logger.warning("Не удалось инициализировать PDF-модуль %s: %s", module_name, e)
                    continue
                if hasattr(module, 'open'):
                    cls._pdf_module = module
                    cls._pdf_module_checked = True
                    return module

            cls._pdf_module_checked = True

        return None

    @classmethod
    def extract(cls, file_path: str, config: Optional[ExtractConfig] = None) -> ExtractResult:
        """
        Определяет расширение файла.
        :param file_path: 
        :param config:
        :return:
        """
        ext: str = os.path.splitext(file_path)[1].lower()
        handler_name = cls._handlers.get(ext)
        active_config = config or cls._default_config
        context = ExtractContext(config=active_config)

        if handler_name is None:
            warning = f'Неподдерживаемое расширение файла: {ext or "<none>"}'
            logger.warning('%s (%s)', warning, file_path)
            return ExtractResult(
                file_path=file_path,
                text='',
                status='unsupported',
                extractor='unsupported',
                warnings=[warning],
            )

        extractor = cls()
        context.handler_name = handler_name
        handler = getattr(extractor, handler_name)

        try:
            text = handler(file_path, context)
        except Exception as exc:
            logger.exception(
                'Экстрактор %s завершился с ошибкой для %s',
                handler_name,
                os.path.basename(file_path),
            )
            return ExtractResult(
                file_path=file_path,
                text='',
                status='failed',
                extractor=handler_name,
                warnings=context.warnings,
                error=str(exc),
            )

        normalized_text = cls._finalize_text(text, context)
        status = cls._resolve_status(normalized_text, context.warnings)

        if status in ('failed', 'partial'):
            logger.warning(
                'Экстрактор %s завершился со статусом %s для %s: %s',
                handler_name,
                status,
                os.path.basename(file_path),
                '; '.join(context.warnings),
            )
        elif status == 'empty':
            logger.info(
                'Экстрактор %s вернул пустой текст для %s',
                handler_name,
                os.path.basename(file_path),
            )

        return ExtractResult(
            file_path=file_path,
            text=normalized_text,
            status=status,
            extractor=handler_name,
            warnings=context.warnings,
        )

    @classmethod
    def extract_text(cls, file_path: str, config: Optional[ExtractConfig] = None) -> str:
        """
        Возвращает только текст без полного объекта результата.
        :param file_path:
        :param config:
        :return:
        """
        return cls.extract(file_path, config=config).text

    @staticmethod
    def _resolve_status(text: str, warnings: Sequence[str]) -> ExtractStatus:
        """
        Определяет итоговый статус извлечения по тексту и предупреждениям.
        :param text:
        :param warnings:
        :return:
        """
        if not text.strip():
            if warnings:
                return 'partial'
            return 'empty'
        if warnings:
            return 'partial'
        return 'success'

    @staticmethod
    def _finalize_text(text: str, context: ExtractContext) -> str:
        """
        Нормализует пробелы и ограничивает итоговую длину текста.
        :param text:
        :param context:
        :return:
        """
        collapsed = ' '.join(text.split())
        if len(collapsed) > context.config.max_chars:
            context.warnings.append(
                f'Текст обрезан до {context.config.max_chars} символов'
            )
            return collapsed[: context.config.max_chars]
        return collapsed

    @staticmethod
    def _normalize_block(value: Any) -> str:
        """
        Приводит блок данных к строке с нормализованными пробелами.
        :param value:
        :return:
        """
        return ' '.join(str(value).split())

    @classmethod
    def _merge_blocks(cls, blocks: Sequence[str]) -> str:
        """
        Объединяет текстовые блоки, удаляя пустые и повторяющиеся значения.
        :param blocks:
        :return:
        """
        unique_blocks: List[str] = []
        seen: set[str] = set()
        for block in blocks:
            normalized = cls._normalize_block(block)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_blocks.append(normalized)
        return ' '.join(unique_blocks)

    @staticmethod
    def _truncate_text(text: str, limit: int, context: ExtractContext, label: str) -> str:
        """
        Обрезает текст до лимита и фиксирует предупреждение в контексте.
        :param text:
        :param limit:
        :param context:
        :param label:
        :return:
        """
        if len(text) > limit:
            context.warnings.append(f'{label} обрезан до {limit} символов')
            return text[:limit]
        return text

    def _extract_pdf(self, file_path: str, context: ExtractContext) -> str:
        """
        Извлекает текст со страниц PDF в пределах настроенного лимита.
        :param file_path:
        :param context:
        :return:
        """
        pdf_module = self._get_pdf_module()
        if pdf_module is None:
            context.warnings.append(
                'PDF-обработка недоступна: не найден совместимый модуль PyMuPDF/fitz'
            )
            return ''

        text_blocks: List[str] = []
        with pdf_module.open(file_path) as doc:
            page_count = len(doc)
            page_limit = min(page_count, context.config.max_pdf_pages)
            if page_count > page_limit:
                context.warnings.append(
                    f'PDF ограничен первыми {page_limit} страницами из {page_count}'
                )

            for page_index in range(page_limit):
                page = doc[page_index]
                page_text = self._normalize_block(page.get_text())
                if page_text:
                    text_blocks.append(page_text)
                    continue

                ocr_text = self._extract_pdf_page_with_ocr(page, context)
                if ocr_text:
                    text_blocks.append(ocr_text)

        return self._merge_blocks(text_blocks)

    def _extract_docx(self, file_path: str, context: ExtractContext) -> str:
        """
        Извлекает текст из абзацев и таблиц документа DOCX.
        :param file_path:
        :param context:
        :return:
        """
        doc = docx.Document(file_path)
        parts: List[str] = [paragraph.text for paragraph in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        return self._merge_blocks(parts)

    def _extract_doc(self, file_path: str, context: ExtractContext) -> str:
        """
        Извлекает текст из DOC через внешнюю утилиту antiword.
        :param file_path:
        :param context:
        :return:
        """
        antiword_path = shutil.which('antiword')
        if antiword_path is None:
            fallback_text = self._extract_doc_as_binary_text(file_path, context)
            if fallback_text:
                context.warnings.append(
                    'antiword не найден; использовано упрощённое извлечение текста из бинарного .doc'
                )
                return fallback_text
            context.warnings.append(
                'Для извлечения .doc требуется antiword; файл помечен как частично поддерживаемый'
            )
            return ''

        completed = subprocess.run(
            [antiword_path, file_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or 'antiword завершился с ошибкой без вывода stderr'
            raise RuntimeError(stderr)
        return completed.stdout

    def _extract_rtf(self, file_path: str, context: ExtractContext) -> str:
        """
        Считывает RTF-файл и преобразует его содержимое в обычный текст.
        :param file_path:
        :param context:
        :return:
        """
        raw_text = self._read_text_with_fallback(file_path, context)
        if not raw_text:
            return ''
        return str(rtf_to_text(raw_text))

    def _extract_table(self, file_path: str, context: ExtractContext) -> str:
        """
        Определяет формат табличного файла и сериализует его в текст.
        :param file_path:
        :param context:
        :return:
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            df = self._read_csv_with_fallback(file_path, context)
        elif ext == '.tsv':
            df = self._read_csv_with_fallback(file_path, context, delimiter='\t')
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_excel(file_path)

        return self._serialize_table(df, context)

    def _read_csv_with_fallback(self, file_path: str, context: ExtractContext, delimiter: str = ',') -> TableFrame:
        """
        Пытается прочитать CSV/TSV в нескольких кодировках.
        :param file_path:
        :param context:
        :param delimiter:
        :return:
        """
        encodings = ('utf-8-sig', 'utf-8', 'cp1251', 'latin-1')
        last_decode_error: Optional[str] = None

        for encoding in encodings:
            try:
                return pd.read_csv(
                    file_path,
                    on_bad_lines='skip',
                    encoding=encoding,
                    sep=delimiter,
                    low_memory=False,
                )
            except UnicodeDecodeError as exc:
                last_decode_error = str(exc)
                continue
            except pd.errors.ParserError:
                raise
            except OSError:
                raise

        if last_decode_error:
            raise UnicodeDecodeError('csv', b'', 0, 1, last_decode_error)
        raise ValueError(f'Не удалось декодировать файл таблицы: {file_path}')

    def _extract_doc_as_binary_text(
        self, file_path: str, context: ExtractContext
    ) -> str:
        """
        Пытается извлечь читаемые строковые фрагменты из бинарного DOC без внешних утилит.
        :param file_path:
        :param context:
        :return:
        """
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
        except OSError as exc:
            context.warnings.append(f'Не удалось прочитать .doc-файл: {exc}')
            return ''

        text_candidates: List[str] = []
        for encoding in ('utf-16le', 'cp1251', 'latin-1'):
            try:
                decoded = raw_data.decode(encoding, errors='ignore')
            except LookupError:
                continue

            fragments = re.findall(r'[\wА-Яа-яЁё.,;:()@/\- ]{4,}', decoded)
            text_candidates.extend(fragment.strip() for fragment in fragments if fragment.strip())

        return self._merge_blocks(text_candidates)

    def _serialize_table(self, df: TableFrame, context: ExtractContext) -> str:
        """
        Преобразует таблицу DataFrame в компактное текстовое представление.
        :param df:
        :param context:
        :return:
        """
        if df.empty:
            return ''

        limited_df = df.head(context.config.max_table_rows).fillna('')
        if len(df.index) > len(limited_df.index):
            context.warnings.append(
                f'Таблица ограничена первыми {len(limited_df.index)} строками из {len(df.index)}'
            )

        columns = [str(column) for column in limited_df.columns]
        lines: List[str] = [f'Столбцы: {", ".join(columns)}']

        for _, row in limited_df.iterrows():
            parts = []
            for column in columns:
                value = self._normalize_block(row[column])
                if value:
                    parts.append(f'{column}: {value}')
            if parts:
                lines.append(' | '.join(parts))

        return '\n'.join(lines)

    def _extract_json(self, file_path: str, context: ExtractContext) -> str:
        """
        Загружает JSON и сериализует его в человекочитаемый текст.
        :param file_path:
        :param context:
        :return:
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            data = json.load(file)
        serialized = json.dumps(data, ensure_ascii=False, indent=2)
        return self._truncate_text(
            serialized, context.config.max_json_chars, context, 'JSON-данные'
        )

    def _extract_html(self, file_path: str, context: ExtractContext) -> str:
        """
        Считывает HTML/XML и извлекает из него видимый текст.
        :param file_path:
        :param context:
        :return:
        """
        raw_text = self._read_text_with_fallback(file_path, context)
        if not raw_text:
            return ''
        soup = BeautifulSoup(raw_text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)

    def _extract_plain_text(self, file_path: str, context: ExtractContext) -> str:
        """
        Считывает обычный текстовый файл с подбором кодировки.
        :param file_path:
        :param context:
        :return:
        """
        return self._read_text_with_fallback(file_path, context)

    def _read_text_with_fallback(self, file_path: str, context: ExtractContext) -> str:
        """
        Пытается прочитать текстовый файл в нескольких распространённых кодировках.
        :param file_path:
        :param context:
        :return:
        """
        last_error: Optional[str] = None
        for encoding in ('utf-8-sig', 'utf-8', 'cp1251', 'latin-1'):
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError as exc:
                last_error = str(exc)
                continue
        if last_error:
            context.warnings.append(
                f'Не удалось корректно декодировать текстовый файл: {last_error}'
            )
        return ''

    def _preprocess_for_ocr(self, image: ImageLike, mode: str) -> ImageLike:
        """
        Подготавливает изображение к OCR в зависимости от сценария обработки.
        :param image:
        :param mode:
        :return:
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if mode in {'document', 'video_frame'}:
            processed = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=31,
                C=10,
            )
        else:
            processed = cv2.equalizeHist(gray)
        return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    def _extract_pdf_page_with_ocr(self, page: Any, context: ExtractContext) -> str:
        """
        Рендерит страницу PDF в изображение и извлекает текст через OCR.
        :param page:
        :param context:
        :return:
        """
        if not hasattr(page, 'get_pixmap'):
            context.warnings.append(
                'OCR для PDF недоступен: модуль PDF не поддерживает рендеринг страниц'
            )
            return ''

        try:
            pixmap = page.get_pixmap(dpi=200)
        except TypeError:
            try:
                pixmap = page.get_pixmap()
            except Exception as exc:
                context.warnings.append(f'Не удалось отрендерить страницу PDF для OCR: {exc}')
                return ''
        except Exception as exc:
            context.warnings.append(f'Не удалось отрендерить страницу PDF для OCR: {exc}')
            return ''

        width = getattr(pixmap, 'width', 0)
        height = getattr(pixmap, 'height', 0)
        channels = 4 if getattr(pixmap, 'alpha', 0) else max(getattr(pixmap, 'n', 3), 1)
        if width <= 0 or height <= 0:
            context.warnings.append('Не удалось определить размер страницы PDF для OCR')
            return ''

        try:
            import numpy as np  # type: ignore
        except Exception as exc:
            context.warnings.append(f'Не удалось загрузить numpy для OCR PDF: {exc}')
            return ''

        try:
            page_image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                height, width, channels
            )
        except ValueError as exc:
            context.warnings.append(f'Не удалось преобразовать страницу PDF в изображение: {exc}')
            return ''

        if channels == 4:
            page_image = cv2.cvtColor(page_image, cv2.COLOR_RGBA2BGR)
        elif channels == 1:
            page_image = cv2.cvtColor(page_image, cv2.COLOR_GRAY2BGR)
        else:
            page_image = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)

        processed = self._preprocess_for_ocr(page_image, mode='document')
        return self._run_ocr(processed, context, rotate_passes=context.config.image_rotate_passes)

    def _run_ocr(
        self, image: ImageLike, context: ExtractContext, rotate_passes: int
    ) -> str:
        """
        Запускает OCR для изображения с дополнительными поворотами кадра.
        :param image:
        :param context:
        :param rotate_passes:
        :return:
        """
        if not context.config.enable_ocr:
            context.warnings.append('OCR отключён в конфигурации')
            return ''

        reader = self._get_reader()
        blocks: List[str] = []
        current_image = image

        for _ in range(max(rotate_passes, 0) + 1):
            result: List[str] = reader.readtext(current_image, detail=0, paragraph=True)
            blocks.extend(result)
            current_image = cv2.rotate(current_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return self._merge_blocks(blocks)

    def _extract_image(self, file_path: str, context: ExtractContext) -> str:
        """
        Загружает изображение, подготавливает его и извлекает текст через OCR.
        :param file_path:
        :param context:
        :return:
        """
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError('OpenCV не удалось загрузить изображение')

        processed = self._preprocess_for_ocr(image, mode='document')
        return self._run_ocr(
            processed, context, rotate_passes=context.config.image_rotate_passes
        )

    def _extract_video(self, file_path: str, context: ExtractContext) -> str:
        """
        Извлекает текст из видео по выбранным кадрам в пределах заданного лимита.
        :param file_path:
        :param context:
        :return:
        """
        if not context.config.enable_ocr:
            context.warnings.append('OCR для видео отключён в конфигурации')
            return ''

        capture = cv2.VideoCapture(file_path)
        if not capture.isOpened():
            raise ValueError('OpenCV не удалось открыть видео')

        fps_raw = capture.get(cv2.CAP_PROP_FPS)
        fps_value = float(fps_raw) if fps_raw and fps_raw > 0 else 24.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = total_frames / fps_value if fps_value > 0 else 0.0
        max_seconds = min(total_seconds, context.config.video_max_seconds)
        frame_step = max(int(round(fps_value / max(context.config.video_fps_sample, 0.1))), 1)

        text_blocks: List[str] = []
        frame_index = 0

        try:
            if total_seconds > context.config.video_max_seconds:
                context.warnings.append(
                    'OCR для видео ограничен первыми '
                    f'{context.config.video_max_seconds:.1f} секундами из '
                    f'{total_seconds:.1f}'
                )

            while True:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success:
                    break

                current_second = frame_index / fps_value if fps_value > 0 else 0.0
                if current_second > max_seconds:
                    break

                processed = self._preprocess_for_ocr(frame, mode='video_frame')
                text = self._run_ocr(processed, context, rotate_passes=0)
                if text:
                    text_blocks.append(text)

                frame_index += frame_step
        finally:
            capture.release()

        return self._merge_blocks(text_blocks)
