import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

try:
    from extractor import ExtractConfig, ExtractResult, FileExtractor
except ImportError:
    from extractor import ExtractConfig, ExtractResult, FileExtractor  # type: ignore

if TYPE_CHECKING:
    try:
        from scanner import PIIAnalyzer
    except ImportError:
        from scanner import PIIAnalyzer

logger = logging.getLogger(__name__)

ANCHOR_CATEGORIES: Tuple[str, ...] = (
    'email',
    'phone',
    'snils',
    'inn',
    'passport_rf',
)

@dataclass
class ScanOutcome:
    file_path: str
    extraction: ExtractResult
    analysis: Optional[Dict[str, Any]]
    anchors: Dict[str, List[str]]


def collect_files(root_dir: str) -> List[str]:
    files: List[str] = []
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


def chunk_files(
    files: Sequence[str], chunk_size: int = 100
) -> List[Tuple[str, ...]]:
    if chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')

    return [
        tuple(files[index : index + chunk_size])
        for index in range(0, len(files), chunk_size)
    ]

def scan_file(
    file_path: str,
    analyzer: Any,
    config: Optional[ExtractConfig] = None,
) -> ScanOutcome:
    extraction = FileExtractor.extract(file_path, config=config)

    if extraction.status in ('failed', 'unsupported'):
        logger.warning(
            'Не удалось извлечь текст из %s [%s]: %s',
            file_path,
            extraction.status,
            extraction.error or '; '.join(extraction.warnings),
        )

    text = str(extraction.text)
    if not text.strip():
        return ScanOutcome(
            file_path=file_path,
            extraction=extraction,
            analysis=None,
            anchors={},
        )

    analysis = analyzer.analyze_text(text)
    analysis['extraction'] = {
        'status': extraction.status,
        'extractor': extraction.extractor,
        'warnings': extraction.warnings,
        'error': extraction.error,
    }

    raw_data = cast(Dict[str, List[str]], analysis.get('raw_data', {}))
    anchors = {
        category: list(raw_data.get(category, []))
        for category in ANCHOR_CATEGORIES
        if raw_data.get(category)
    }

    return ScanOutcome(
        file_path=file_path,
        extraction=extraction,
        analysis=analysis,
        anchors=anchors,
    )


def process_batch(
    batch: Sequence[str],
    config: Optional[ExtractConfig] = None,
    analyzer_factory: Optional[Callable[[], Any]] = None,
) -> List[ScanOutcome]:
    if analyzer_factory is None:
        analyzer_factory = _build_analyzer

    analyzer = analyzer_factory()
    results: List[ScanOutcome] = []

    for file_path in batch:
        try:
            results.append(scan_file(file_path, analyzer, config=config))
        except Exception as exc:
            logger.exception('Неожиданная ошибка обработки файла %s', file_path)
            results.append(
                ScanOutcome(
                    file_path=file_path,
                    extraction=ExtractResult(
                        file_path=file_path,
                        text='',
                        status='failed',
                        extractor='batch_router',
                        warnings=[],
                        error=str(exc),
                    ),
                    analysis=None,
                    anchors={},
                )
            )

    return results


def _build_analyzer() -> Any:
    try:
        from scanner import PIIAnalyzer
    except ImportError:
        from scanner import PIIAnalyzer
    return PIIAnalyzer()
