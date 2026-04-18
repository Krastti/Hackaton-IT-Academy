from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests._test_support import make_smoke_dir, prepare_test_env, print_test_report

prepare_test_env()

from src.app import PIIController
from src.batch_router import ScanOutcome
from src.extractor import ExtractResult


def _build_scan_outcome(file_path: str) -> ScanOutcome:
    name = Path(file_path).stem
    if name == 'd':
        return ScanOutcome(
            file_path=file_path,
            extraction=ExtractResult(
                file_path=file_path,
                text='',
                status='unsupported',
                extractor='fake',
                warnings=['skip'],
            ),
            analysis=None,
            anchors={},
        )

    raw_data = {
        'email': [f'{name}@example.com'],
        'phone': ['79991234567'] if name in {'a', 'b'} else [],
        'snils': [],
        'inn': [],
        'passport_rf': [],
    }
    return ScanOutcome(
        file_path=file_path,
        extraction=ExtractResult(
            file_path=file_path,
            text=f'text for {name}',
            status='success',
            extractor='fake',
        ),
        analysis={
            'raw_data': raw_data,
            'stats': {
                'common_pii': len(raw_data['email']) + len(raw_data['phone']),
                'gov_ids': 0,
                'payment_info': 0,
                'special_pii': 0,
            },
            'extraction': {
                'status': 'success',
                'extractor': 'fake',
                'warnings': [],
                'error': None,
            },
        },
        anchors={
            'email': raw_data['email'],
            'phone': raw_data['phone'],
        },
    )


def _run_parallel_scan(tmp_path: Path) -> tuple[PIIController, PIIController, dict[str, dict[str, object]]]:
    dataset_dir = tmp_path / 'dataset'
    dataset_dir.mkdir(exist_ok=True)
    files = [dataset_dir / name for name in ('a.txt', 'b.txt', 'c.txt', 'd.txt')]
    for file_path in files:
        file_path.write_text('placeholder', encoding='utf-8')

    outputs: dict[str, dict[str, object]] = {}

    def fake_collect_files(root_dir: str) -> list[str]:
        return [str(file_path) for file_path in files]

    def fake_process_batch(batch: tuple[str, ...], config: object) -> list[ScanOutcome]:
        return [_build_scan_outcome(file_path) for file_path in batch]

    def fake_write_reports(self, file_registry, groups):
        outputs[self.output_prefix] = {
            'file_registry': dict(file_registry),
            'groups': {frozenset(group) for group in groups},
        }
        return []

    with patch('src.app.collect_files', fake_collect_files), patch(
        'src.app.process_batch', fake_process_batch
    ), patch('src.reporter.PIIReporter.write_reports', fake_write_reports):
        single = PIIController(str(dataset_dir), output_prefix='single', chunk_size=2, max_workers=1)
        single.run_scan()

        multi = PIIController(str(dataset_dir), output_prefix='multi', chunk_size=2, max_workers=3)
        multi.run_scan()

    return single, multi, outputs


def test_run_scan_matches_results_for_single_and_multi_worker(tmp_path: Path) -> None:
    single, multi, outputs = _run_parallel_scan(tmp_path)

    assert set(single.extraction_registry) == set(multi.extraction_registry)
    assert single.file_registry == multi.file_registry
    assert {key: set(values) for key, values in single.anchor_index.items()} == {
        key: set(values) for key, values in multi.anchor_index.items()
    }
    assert outputs['single']['file_registry'] == outputs['multi']['file_registry']
    assert outputs['single']['groups'] == outputs['multi']['groups']


if __name__ == '__main__':
    print_test_report(
        'Отчёт по тестам параллельного сканирования',
        [
            (
                'Сравнение однопоточного и многопоточного сканирования',
                lambda: test_run_scan_matches_results_for_single_and_multi_worker(
                    make_smoke_dir('parallel-scan')
                ),
            )
        ],
    )
