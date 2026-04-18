from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests._test_support import make_smoke_dir, prepare_test_env, print_test_report

prepare_test_env()

from src.extractor import ExtractConfig, FileExtractor


class _FakeDataFrame:
    def __init__(self) -> None:
        self.columns = ['name', 'email']
        self._rows = [
            {'name': 'Ivan', 'email': 'ivan@example.com'},
            {'name': 'Anna', 'email': 'anna@example.com'},
        ]

    @property
    def empty(self) -> bool:
        return False

    @property
    def index(self) -> range:
        return range(len(self._rows))

    def head(self, count: int) -> '_FakeDataFrame':
        frame = _FakeDataFrame()
        frame._rows = self._rows[:count]
        return frame

    def fillna(self, value: str) -> '_FakeDataFrame':
        return self

    def iterrows(self):
        for index, row in enumerate(self._rows):
            yield index, row


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
    assert any('JSON-данные обрезан до 40 символов' in warning for warning in result.warnings)


def test_table_extraction_limits_rows_and_serializes_columns(tmp_path: Path) -> None:
    sample = tmp_path / 'sample.csv'
    sample.write_text('name,email\nIvan,ivan@example.com\nAnna,anna@example.com\n', encoding='utf-8')

    with patch('src.extractor.pd.read_csv', lambda *args, **kwargs: _FakeDataFrame()):
        result = FileExtractor.extract(
            str(sample),
            config=ExtractConfig(max_table_rows=1, max_chars=500),
        )

    assert result.status == 'partial'
    assert 'Столбцы: name, email' in result.text
    assert 'name: Ivan | email: ivan@example.com' in result.text
    assert 'anna@example.com' not in result.text
    assert any('Таблица ограничена первыми 1 строками из 2' in warning for warning in result.warnings)


def test_doc_without_antiword_is_marked_partial(tmp_path: Path) -> None:
    sample = tmp_path / 'legacy.doc'
    sample.write_bytes(b'not-a-real-doc')

    with patch('src.extractor.shutil.which', lambda _: None), patch.object(
        FileExtractor,
        '_extract_doc_as_binary_text',
        lambda self, file_path, context: '',
    ):
        result = FileExtractor.extract(str(sample))

    assert result.status == 'partial'
    assert result.text == ''
    assert any('antiword' in warning for warning in result.warnings)


def test_image_ocr_can_be_disabled(tmp_path: Path) -> None:
    sample = tmp_path / 'sample.png'
    sample.write_bytes(b'placeholder')

    with patch('src.extractor.cv2.imread', lambda _: object()), patch.object(
        FileExtractor,
        '_get_reader',
        classmethod(lambda cls: (_ for _ in ()).throw(AssertionError('OCR reader should not be used'))),
    ):
        result = FileExtractor.extract(
            str(sample),
            config=ExtractConfig(enable_ocr=False),
        )

    assert result.status == 'partial'
    assert result.text == ''
    assert any('OCR отключён' in warning for warning in result.warnings)


if __name__ == '__main__':
    print_test_report(
        'Отчёт по тестам экстрактора',
        [
            ('Неподдерживаемое расширение', lambda: test_unsupported_extension_returns_explicit_status(make_smoke_dir('extractor-unsupported'))),
            ('Обрезка JSON', lambda: test_json_truncation_is_reported(make_smoke_dir('extractor-json'))),
            ('Ограничение строк таблицы', lambda: test_table_extraction_limits_rows_and_serializes_columns(make_smoke_dir('extractor-table'))),
            ('DOC без antiword', lambda: test_doc_without_antiword_is_marked_partial(make_smoke_dir('extractor-doc'))),
            ('Отключение OCR для изображения', lambda: test_image_ocr_can_be_disabled(make_smoke_dir('extractor-image'))),
        ],
    )
