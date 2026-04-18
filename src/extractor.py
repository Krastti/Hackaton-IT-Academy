import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Sequence

import cv2  # type: ignore
import docx  # type: ignore
import easyocr  # type: ignore
import fitz  # PyMuPDF  # type: ignore
import pandas as pd  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
from striprtf.striprtf import rtf_to_text  # type: ignore


logger = logging.getLogger(__name__)

ExtractStatus = Literal['success', 'empty', 'unsupported', 'failed', 'partial']
ImageLike = Any
TableFrame = Any
ExtractorHandler = Callable[['FileExtractor', str, 'ExtractContext'], str]


@dataclass(frozen=True)
class ExtractConfig:
    enable_ocr: bool = True
    image_rotate_passes: int = 1
    video_max_seconds: float = 120.0
    video_fps_sample: float = 2.0
    max_pdf_pages: int = 200
    max_table_rows: int = 1000
    max_json_chars: int = 200000
    max_chars: int = 200000


@dataclass
class ExtractResult:
    file_path: str
    text: str
    status: ExtractStatus
    extractor: str
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ExtractContext:
    config: ExtractConfig
    warnings: List[str] = field(default_factory=list)
    handler_name: str = ''


class FileExtractor:
    _reader: ClassVar[Optional[easyocr.Reader]] = None
    _default_config: ClassVar[ExtractConfig] = ExtractConfig()
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
        if cls._reader is None:
            cls._reader = easyocr.Reader(['ru', 'en'], gpu=False)
        return cls._reader

    @classmethod
    def extract(
        cls, file_path: str, config: Optional[ExtractConfig] = None
    ) -> ExtractResult:
        ext: str = os.path.splitext(file_path)[1].lower()
        handler_name = cls._handlers.get(ext)
        active_config = config or cls._default_config
        context = ExtractContext(config=active_config)

        if handler_name is None:
            warning = f'Unsupported file extension: {ext or "<none>"}'
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
                'Extractor %s failed for %s', handler_name, os.path.basename(file_path)
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
                'Extractor %s completed with status %s for %s: %s',
                handler_name,
                status,
                os.path.basename(file_path),
                '; '.join(context.warnings),
            )
        elif status == 'empty':
            logger.info(
                'Extractor %s returned empty text for %s',
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
        return cls.extract(file_path, config=config).text

    @staticmethod
    def _resolve_status(text: str, warnings: Sequence[str]) -> ExtractStatus:
        if not text.strip():
            if warnings:
                return 'partial'
            return 'empty'
        if warnings:
            return 'partial'
        return 'success'

    @staticmethod
    def _finalize_text(text: str, context: ExtractContext) -> str:
        collapsed = ' '.join(text.split())
        if len(collapsed) > context.config.max_chars:
            context.warnings.append(
                f'Text truncated to {context.config.max_chars} characters'
            )
            return collapsed[: context.config.max_chars]
        return collapsed

    @staticmethod
    def _normalize_block(value: Any) -> str:
        return ' '.join(str(value).split())

    @classmethod
    def _merge_blocks(cls, blocks: Sequence[str]) -> str:
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
        if len(text) > limit:
            context.warnings.append(f'{label} truncated to {limit} characters')
            return text[:limit]
        return text

    def _extract_pdf(self, file_path: str, context: ExtractContext) -> str:
        text_blocks: List[str] = []
        with fitz.open(file_path) as doc:
            page_count = len(doc)
            page_limit = min(page_count, context.config.max_pdf_pages)
            if page_count > page_limit:
                context.warnings.append(
                    f'PDF limited to first {page_limit} pages out of {page_count}'
                )

            for page_index in range(page_limit):
                text_blocks.append(doc[page_index].get_text())

        return self._merge_blocks(text_blocks)

    def _extract_docx(self, file_path: str, context: ExtractContext) -> str:
        doc = docx.Document(file_path)
        parts: List[str] = [paragraph.text for paragraph in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        return self._merge_blocks(parts)

    def _extract_doc(self, file_path: str, context: ExtractContext) -> str:
        antiword_path = shutil.which('antiword')
        if antiword_path is None:
            context.warnings.append(
                '.doc extraction requires antiword; file marked as partially supported'
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
            stderr = completed.stderr.strip() or 'antiword failed without stderr'
            raise RuntimeError(stderr)
        return completed.stdout

    def _extract_rtf(self, file_path: str, context: ExtractContext) -> str:
        raw_text = self._read_text_with_fallback(file_path, context)
        if not raw_text:
            return ''
        return str(rtf_to_text(raw_text))

    def _extract_table(self, file_path: str, context: ExtractContext) -> str:
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

    def _read_csv_with_fallback(
        self, file_path: str, context: ExtractContext, delimiter: str = ','
    ) -> TableFrame:
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
        raise ValueError(f'Unable to decode table file: {file_path}')

    def _serialize_table(self, df: TableFrame, context: ExtractContext) -> str:
        if df.empty:
            return ''

        limited_df = df.head(context.config.max_table_rows).fillna('')
        if len(df.index) > len(limited_df.index):
            context.warnings.append(
                f'Table limited to first {len(limited_df.index)} rows out of {len(df.index)}'
            )

        columns = [str(column) for column in limited_df.columns]
        lines: List[str] = [f'Columns: {", ".join(columns)}']

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
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            data = json.load(file)
        serialized = json.dumps(data, ensure_ascii=False, indent=2)
        return self._truncate_text(
            serialized, context.config.max_json_chars, context, 'JSON payload'
        )

    def _extract_html(self, file_path: str, context: ExtractContext) -> str:
        raw_text = self._read_text_with_fallback(file_path, context)
        if not raw_text:
            return ''
        soup = BeautifulSoup(raw_text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)

    def _extract_plain_text(self, file_path: str, context: ExtractContext) -> str:
        return self._read_text_with_fallback(file_path, context)

    def _read_text_with_fallback(self, file_path: str, context: ExtractContext) -> str:
        last_error: Optional[str] = None
        for encoding in ('utf-8-sig', 'utf-8', 'cp1251', 'latin-1'):
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError as exc:
                last_error = str(exc)
                continue
        if last_error:
            context.warnings.append(f'Unable to decode text file cleanly: {last_error}')
        return ''

    def _preprocess_for_ocr(self, image: ImageLike, mode: str) -> ImageLike:
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

    def _run_ocr(
        self, image: ImageLike, context: ExtractContext, rotate_passes: int
    ) -> str:
        if not context.config.enable_ocr:
            context.warnings.append('OCR disabled by configuration')
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
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError('OpenCV failed to load image')

        processed = self._preprocess_for_ocr(image, mode='document')
        return self._run_ocr(
            processed, context, rotate_passes=context.config.image_rotate_passes
        )

    def _extract_video(self, file_path: str, context: ExtractContext) -> str:
        if not context.config.enable_ocr:
            context.warnings.append('Video OCR disabled by configuration')
            return ''

        capture = cv2.VideoCapture(file_path)
        if not capture.isOpened():
            raise ValueError('OpenCV failed to open video')

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
                    'Video OCR limited to first '
                    f'{context.config.video_max_seconds:.1f} seconds out of '
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
