import os
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Set, cast
from collections import defaultdict

from tqdm import tqdm
try:
    from extractor import ExtractConfig, ExtractResult
    from reporter import PIIReporter
    from batch_router import collect_files, chunk_files, process_batch, ScanOutcome
except ImportError:
    from extractor import ExtractConfig, ExtractResult  # type: ignore
    from reporter import PIIReporter  # type: ignore
    from batch_router import collect_files, chunk_files, process_batch, ScanOutcome  # type: ignore

logger = logging.getLogger(__name__)

class PIIController:
    def __init__(
        self,
        target_dir: str,
        output_prefix: str = 'report',
        extract_config: ExtractConfig | None = None,
        chunk_size: int = 100,
        max_workers: int | None = None,
    ) -> None:
        self.target_dir: str = target_dir
        self.output_prefix: str = output_prefix
        self.extract_config: ExtractConfig = extract_config or ExtractConfig()
        if chunk_size <= 0:
            raise ValueError('chunk_size must be a positive integer')
        self.chunk_size: int = chunk_size

        resolved_workers = max_workers if max_workers is not None else min(4, os.cpu_count() or 1)
        if resolved_workers <= 0:
            raise ValueError('max_workers must be a positive integer')
        self.max_workers: int = resolved_workers
        self.reporter: PIIReporter = PIIReporter(output_prefix)

        # {путь: {stats, raw_data}}
        self.file_registry: Dict[str, Dict[str, Any]] = {}
        self.extraction_registry: Dict[str, ExtractResult] = {}
        # {нормализованное_значение: {набор_путей}}
        self.anchor_index: Dict[str, Set[str]] = defaultdict(set)

    def _find_groups(self) -> List[Set[str]]:
        """Алгоритм поиска связанных компонентов (Union-Find по якорям)."""
        visited: Set[str] = set()
        groups: List[Set[str]] = []
        all_files: List[str] = list(self.file_registry.keys())

        for file_path in all_files:
            if file_path not in visited:
                current_group: Set[str] = set()
                queue: List[str] = [file_path]
                visited.add(file_path)

                while queue:
                    node: str = queue.pop(0)
                    current_group.add(node)
                    node_data: Dict[str, Any] = self.file_registry[node]
                    raw_data: Dict[str, List[str]] = cast(
                        Dict[str, List[str]], node_data.get('raw_data', {})
                    )
                    for category in ['email', 'phone', 'snils', 'inn', 'passport_rf']:
                        for value in raw_data.get(category, []):
                            for linked_file in self.anchor_index.get(value, set()):
                                if linked_file not in visited:
                                    visited.add(linked_file)
                                    queue.append(linked_file)

                groups.append(current_group)
        return groups

    def _merge_outcome(self, outcome: ScanOutcome) -> None:
        f_path = outcome.file_path
        self.extraction_registry[f_path] = outcome.extraction

        if outcome.analysis is None:
            return

        self.file_registry[f_path] = outcome.analysis
        for category, values in outcome.anchors.items():
            for value in values:
                self.anchor_index[value].add(f_path)

    def run_scan(self) -> None:
        # ----------------------------------------------------------------
        # Шаг 1: сканирование
        # ----------------------------------------------------------------
        print('[*] Шаг 1: Сканирование и сбор якорей...')
        scan_started_at = time.perf_counter()
        all_files = collect_files(self.target_dir)
        batches = chunk_files(all_files, chunk_size=self.chunk_size)

        logger.info(
            'Найдено %s файлов для обработки. Батчей: %s. Размер батча: %s. Потоков: %s',
            len(all_files),
            len(batches),
            self.chunk_size,
            self.max_workers,
        )

        future_to_batch_index: Dict[Future[List[ScanOutcome]], int] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for batch_index, batch in enumerate(batches, start=1):
                logger.info(
                    'Запуск батча %s/%s (%s файлов)',
                    batch_index,
                    len(batches),
                    len(batch),
                )
                future = executor.submit(process_batch, batch, self.extract_config)
                future_to_batch_index[future] = batch_index

            for future in tqdm(
                as_completed(future_to_batch_index),
                total=len(future_to_batch_index),
                desc='Обработка батчей',
                unit='батч',
            ):
                batch_index = future_to_batch_index[future]
                batch_outcomes = future.result()

                status_counts: Dict[str, int] = defaultdict(int)
                for outcome in batch_outcomes:
                    self._merge_outcome(outcome)
                    status_counts[outcome.extraction.status] += 1

                logger.info(
                    'Завершён батч %s/%s: success=%s partial=%s empty=%s failed=%s unsupported=%s',
                    batch_index,
                    len(batches),
                    status_counts.get('success', 0),
                    status_counts.get('partial', 0),
                    status_counts.get('empty', 0),
                    status_counts.get('failed', 0),
                    status_counts.get('unsupported', 0),
                )

        logger.info(
            'Этап сканирования завершён за %.2f сек.',
            time.perf_counter() - scan_started_at,
        )

        # ----------------------------------------------------------------
        # Шаг 2: группировка
        # ----------------------------------------------------------------
        print('[*] Шаг 2: Группировка файлов и классификация...')
        groups: List[Set[str]] = self._find_groups()

        # ----------------------------------------------------------------
        # Шаг 3: запись отчётов
        # ----------------------------------------------------------------
        self.reporter.write_reports(self.file_registry, groups)

        print(f'[+] Отчёты: {self.output_prefix}.csv / .json / .md')


if __name__ == '__main__':
    app = PIIController('../TestDataset', 'report')
    app.run_scan()
