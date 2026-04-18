import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extractor import ExtractConfig, FileExtractor


def test_unsupported_extension_returns_explicit_status(tmp_path: Path) -> None:
    sample = tmp_path / 'sample.bin'
    sample.write_bytes(b'123')

    result = FileExtractor.extract(str(sample))

    assert result.status == 'unsupported'
    assert result.text == ''
    assert result.extractor == 'unsupported'
    assert result.warnings


def test_json_truncation_is_reported(tmp_path: Path) -> None:
    sample = tmp_path / 'sample.json'
    sample.write_text('{"value": "' + ('x' * 200) + '"}', encoding='utf-8')

    result = FileExtractor.extract(
        str(sample),
        config=ExtractConfig(max_json_chars=40, max_chars=500),
    )

    assert result.status == 'partial'
    assert len(result.text) <= 40
    assert any('JSON payload truncated' in warning for warning in result.warnings)


def test_table_extraction_limits_rows_and_serializes_columns(tmp_path: Path) -> None:
    sample = tmp_path / 'sample.csv'
    sample.write_text('name,email\nIvan,ivan@example.com\nAnna,anna@example.com\n', encoding='utf-8')

    result = FileExtractor.extract(
        str(sample),
        config=ExtractConfig(max_table_rows=1, max_chars=500),
    )

    assert result.status == 'partial'
    assert 'Columns: name, email' in result.text
    assert 'name: Ivan | email: ivan@example.com' in result.text
    assert 'anna@example.com' not in result.text
    assert any('Table limited to first 1 rows out of 2' in warning for warning in result.warnings)


def test_doc_without_antiword_is_marked_partial(tmp_path: Path, monkeypatch) -> None:
    sample = tmp_path / 'legacy.doc'
    sample.write_bytes(b'not-a-real-doc')
    monkeypatch.setattr('extractor.shutil.which', lambda _: None)

    result = FileExtractor.extract(str(sample))

    assert result.status == 'partial'
    assert result.text == ''
    assert any('antiword' in warning for warning in result.warnings)


def test_image_ocr_can_be_disabled(tmp_path: Path, monkeypatch) -> None:
    sample = tmp_path / 'sample.png'
    sample.write_bytes(b'placeholder')

    monkeypatch.setattr('extractor.cv2.imread', lambda _: np.zeros((4, 4, 3), dtype=np.uint8))
    monkeypatch.setattr(
        FileExtractor,
        '_get_reader',
        classmethod(lambda cls: (_ for _ in ()).throw(AssertionError('OCR reader should not be used'))),
    )

    result = FileExtractor.extract(
        str(sample),
        config=ExtractConfig(enable_ocr=False),
    )

    assert result.status == 'partial'
    assert result.text == ''
    assert any('OCR disabled' in warning for warning in result.warnings)
