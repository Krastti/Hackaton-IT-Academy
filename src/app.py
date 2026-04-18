import os
import logging
from typing import Dict, Any, List, Set, cast
from collections import defaultdict

from tqdm import tqdm
from extractor import ExtractResult, FileExtractor  # type: ignore
from scanner import PIIAnalyzer  # type: ignore
from reporter import PIIReporter  # type: ignore


logger = logging.getLogger(__name__)


class PIIController:
    def __init__(self, target_dir: str, output_prefix: str = 'report') -> None:
        self.target_dir: str = target_dir
        self.output_prefix: str = output_prefix
        self.analyzer: PIIAnalyzer = PIIAnalyzer()
        self.extractor: FileExtractor = FileExtractor()
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

    def run_scan(self) -> None:
        # ----------------------------------------------------------------
        # Шаг 1: сканирование
        # ----------------------------------------------------------------
        print('[*] Шаг 1: Сканирование и сбор якорей...')
        all_files: List[str] = []
        for root, _, files in os.walk(self.target_dir):
            for f in files:
                all_files.append(os.path.join(root, f))

        for f_path in tqdm(all_files, desc='Сканирование', unit='файл'):
            extraction: ExtractResult = self.extractor.extract(f_path)
            self.extraction_registry[f_path] = extraction

            if extraction.status in ('failed', 'unsupported'):
                logger.warning(
                    'Не удалось извлечь текст из %s [%s]: %s',
                    f_path,
                    extraction.status,
                    extraction.error or '; '.join(extraction.warnings),
                )

            text: str = str(extraction.text)
            if not text.strip():
                continue
            res: Dict[str, Any] = self.analyzer.analyze_text(text)
            res['extraction'] = {
                'status': extraction.status,
                'extractor': extraction.extractor,
                'warnings': extraction.warnings,
                'error': extraction.error,
            }
            self.file_registry[f_path] = res

            raw_data: Dict[str, List[str]] = cast(
                Dict[str, List[str]], res.get('raw_data', {})
            )
            for cat in ['email', 'phone', 'snils', 'inn', 'passport_rf']:
                for val in raw_data.get(cat, []):
                    self.anchor_index[val].add(f_path)

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
    app = PIIController('./test_dataset', 'report')
    app.run_scan()
