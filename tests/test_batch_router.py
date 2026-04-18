from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests._test_support import prepare_test_env, print_test_report

prepare_test_env()

from src.batch_router import chunk_files, process_batch
from src.extractor import ExtractResult


def test_chunk_files_handles_boundaries() -> None:
    files = [f'file_{index}' for index in range(250)]

    assert chunk_files([], chunk_size=100) == []
    assert chunk_files(['single'], chunk_size=100) == [('single',)]
    assert len(chunk_files(files[:100], chunk_size=100)) == 1
    assert len(chunk_files(files[:101], chunk_size=100)) == 2
    assert [len(batch) for batch in chunk_files(files, chunk_size=100)] == [100, 100, 50]


def test_chunk_files_preserves_order_inside_batches() -> None:
    files = ['a', 'b', 'c', 'd', 'e']

    assert chunk_files(files, chunk_size=2) == [('a', 'b'), ('c', 'd'), ('e',)]


def test_process_batch_returns_failed_outcome_when_scan_file_raises() -> None:
    scanned: list[str] = []

    def fake_scan_file(file_path: str, analyzer: object, config: object = None) -> object:
        scanned.append(file_path)
        if file_path == 'broken.txt':
            raise RuntimeError('boom')
        return object()

    with patch('src.batch_router.scan_file', fake_scan_file):
        outcomes = process_batch(
            ['ok.txt', 'broken.txt', 'ok2.txt'],
            analyzer_factory=lambda: object(),
        )

    assert scanned == ['ok.txt', 'broken.txt', 'ok2.txt']
    assert len(outcomes) == 3
    assert outcomes[1].file_path == 'broken.txt'
    assert outcomes[1].extraction.status == 'failed'
    assert outcomes[1].extraction.error == 'boom'


def test_process_batch_returns_results_for_every_file() -> None:
    def fake_scan_file(file_path: str, analyzer: object, config: object = None) -> object:
        return type(
            'Outcome',
            (),
            {
                'file_path': file_path,
                'extraction': ExtractResult(
                    file_path=file_path,
                    text='text',
                    status='success',
                    extractor='fake',
                ),
                'analysis': {'raw_data': {}, 'stats': {}},
                'anchors': {},
            },
        )()

    with patch('src.batch_router.scan_file', fake_scan_file):
        outcomes = process_batch(
            ['1.txt', '2.txt', '3.txt'],
            analyzer_factory=lambda: object(),
        )

    assert [outcome.file_path for outcome in outcomes] == ['1.txt', '2.txt', '3.txt']


if __name__ == '__main__':
    print_test_report(
        'Отчёт по тестам маршрутизации батчей',
        [
            ('Границы разбиения файлов', test_chunk_files_handles_boundaries),
            ('Порядок файлов внутри батчей', test_chunk_files_preserves_order_inside_batches),
            ('Ошибка scan_file превращается в failed', test_process_batch_returns_failed_outcome_when_scan_file_raises),
            ('Результат возвращается для каждого файла', test_process_batch_returns_results_for_every_file),
        ],
    )
