import os
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Set, cast
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from extractor import FileExtractor
from analyzer import PIIAnalyzer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

LARGE_VOLUME_THRESHOLD = 1000

CATEGORY_LABELS = {
    'fio': 'ФИО', 'email': 'Email', 'phone': 'Телефон', 'address': 'Адрес',
    'passport': 'Паспорт', 'snils': 'СНИЛС', 'inn': 'ИНН',
    'driver_license': 'Водительское удостоверение', 'bank_card': 'Банковская карта',
    'bik': 'БИК', 'cvv': 'CVV/CVC', 'special_pii': 'Спецкатегории / Биометрия',
}

META_CATEGORY_LABELS = {
    'common_pii': 'Общие ПДн', 'gov_ids': 'Гос. идентификаторы',
    'payment_info': 'Платёжные данные', 'special_pii': 'Спец. ПДн / Биометрия'
}

UZ_RECOMMENDATIONS = {
    'УЗ-1': 'ВЫСОКИЙ РИСК. Наличие спецкатегорий или биометрии.',
    'УЗ-2': 'ПОВЫШЕННЫЙ РИСК. Платёжные данные или объёмные госидентификаторы.',
    'УЗ-3': 'СРЕДНИЙ РИСК. Ограниченный объём госидентификаторов.',
    'УЗ-4': 'БАЗОВЫЙ УРОВЕНЬ. Небольшой объём обычных ПДн.',
    'Безопасно': 'ПДн не обнаружены.'
}


class PIIController:
    def __init__(self, target_dir: str, output_prefix: str = 'report'):
        self.target_dir = target_dir
        self.output_prefix = output_prefix
        self.analyzer = PIIAnalyzer()
        self.extractor = FileExtractor()
        self.file_registry = {}
        self.anchor_index = defaultdict(set)

    def _determine_uz(self, stats: Dict[str, int]) -> str:
        if stats.get('special_pii', 0) > 0: return 'УЗ-1'
        if stats.get('payment_info', 0) > LARGE_VOLUME_THRESHOLD or stats.get('gov_ids',
                                                                              0) > LARGE_VOLUME_THRESHOLD: return 'УЗ-2'
        if (0 < stats.get('gov_ids', 0) <= LARGE_VOLUME_THRESHOLD) or stats.get('common_pii',
                                                                                0) > LARGE_VOLUME_THRESHOLD: return 'УЗ-3'
        if 0 < stats.get('common_pii', 0) <= LARGE_VOLUME_THRESHOLD: return 'УЗ-4'
        return 'Безопасно'

    def _find_groups(self) -> List[Set[str]]:
        visited, groups = set(), []
        for file_path in self.file_registry.keys():
            if file_path not in visited:
                current_group, queue = set(), [file_path]
                visited.add(file_path)
                while queue:
                    node = queue.pop(0)
                    current_group.add(node)
                    raw_data = self.file_registry[node].get('raw_data', {})
                    for category in ['email', 'phone', 'snils', 'inn', 'passport']:
                        for value in raw_data.get(category, []):
                            for linked_file in self.anchor_index.get(value, set()):
                                if linked_file not in visited:
                                    visited.add(linked_file)
                                    queue.append(linked_file)
                groups.append(current_group)
        return groups

    def run_scan(self) -> None:
        print('[*] Шаг 1: Сканирование...')
        all_files = [os.path.join(r, f) for r, _, fs in os.walk(self.target_dir) for f in fs]

        for f_path in tqdm(all_files, desc='Обработка', unit='ф'):
            text = str(self.extractor.extract_text(f_path))
            res = self.analyzer.analyze_text(text)
            self.file_registry[f_path] = res

            raw_data = res.get('raw_data', {})
            for cat in ['email', 'phone', 'snils', 'inn', 'passport']:
                for val in raw_data.get(cat, []):
                    self.anchor_index[val].add(f_path)

        print('[*] Шаг 2: Группировка...')
        groups = self._find_groups()
        report_rows = []

        for group_id, group_files in enumerate(groups, start=1):
            group_raw_union = defaultdict(set)
            group_special = 0

            for f_path in group_files:
                f_res = self.file_registry[f_path]
                for cat, items in f_res.get('raw_data', {}).items():
                    group_raw_union[cat].update(items)
                group_special += int(f_res.get('stats', {}).get('special_pii', 0))

            group_stats = {
                'common_pii': sum(len(group_raw_union[k]) for k in ('fio', 'email', 'phone', 'address')),
                'gov_ids': sum(len(group_raw_union[k]) for k in ('passport', 'snils', 'inn', 'driver_license')),
                'payment_info': sum(len(group_raw_union[k]) for k in ('bank_card', 'bik', 'cvv')),
                'special_pii': group_special,
            }

            uz_level = self._determine_uz(group_stats)
            group_categories = {k for k, v in group_stats.items() if v > 0}
            group_counts = {cat: len(vals) for cat, vals in group_raw_union.items() if vals}

            for f_path in group_files:
                report_rows.append({
                    'path': f_path, 'group_id': f'Группа_{group_id}',
                    'categories': [META_CATEGORY_LABELS.get(c, c) for c in sorted(group_categories)],
                    'counts': {CATEGORY_LABELS.get(k, k): v for k, v in group_counts.items() if v > 0},
                    'uz': uz_level, 'file_format': Path(f_path).suffix,
                    'recommendation': UZ_RECOMMENDATIONS.get(uz_level, ''),
                })

        print(f'[+] Отчёты: {self.output_prefix}.csv / .json / .md')
        self._write_csv(report_rows)
        self._write_json(report_rows)
        self._write_markdown(report_rows, groups)

    def _write_csv(self, rows):
        with open(self.output_prefix + '.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['Путь', 'Группа', 'Категории_ПДн', 'Находки', 'УЗ', 'Формат', 'Рекомендации'])
            for r in rows:
                writer.writerow([r['path'], r['group_id'], ', '.join(r['categories']),
                                 '; '.join(f"{k}: {v}" for k, v in r['counts'].items()),
                                 r['uz'], r['file_format'], r['recommendation']])

    def _write_json(self, rows):
        with open(self.output_prefix + '.json', 'w', encoding='utf-8') as f:
            json.dump({'total': len(rows), 'results': rows}, f, ensure_ascii=False, indent=2)

    def _write_markdown(self, rows, groups):
        with open(self.output_prefix + '.md', 'w', encoding='utf-8') as f:
            f.write(f"# Отчёт PII-сканера\n**Файлов:** {len(rows)}\n\n")
            for r in rows[:100]:  # Ограничиваем вывод в MD для гигантских датасетов
                f.write(f"- `{Path(r['path']).name}` | {r['uz']} | {', '.join(r['categories'])}\n")


if __name__ == '__main__':
    app = PIIController('../test_dataset', 'report')
    app.run_scan()