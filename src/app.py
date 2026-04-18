import os
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, cast
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm  # type: ignore
from extractor import FileExtractor  # type: ignore
from analyzer import PIIAnalyzer  # type: ignore

# Порог «большого объёма» по ТЗ: «десятки тысяч записей»
LARGE_VOLUME_THRESHOLD: int = 1000

# Человекочитаемые названия категорий
CATEGORY_LABELS: Dict[str, str] = {
    'fio':            'ФИО',
    'email':          'Email',
    'phone':          'Телефон',
    'address':        'Адрес',
    'passport_rf':    'Паспорт РФ',
    'snils':          'СНИЛС',
    'inn':            'ИНН',
    'driver_license': 'Водительское удостоверение',
    'bank_card':      'Банковская карта',
    'bik':            'БИК',
    'cvv':            'CVV/CVC',
    'special_pii':    'Спецкатегории / Биометрия',
}

# Рекомендации по обработке в зависимости от УЗ
UZ_RECOMMENDATIONS: Dict[str, str] = {
    'УЗ-1': (
        'ВЫСОКИЙ РИСК. Требуется немедленное выполнение требований 152-ФЗ: '
        'назначить ответственного за обработку ПДн, провести инвентаризацию, '
        'ограничить доступ, организовать шифрование, уведомить Роскомнадзор.'
    ),
    'УЗ-2': (
        'ПОВЫШЕННЫЙ РИСК. Платёжные данные или объёмные госидентификаторы. '
        'Необходимо внедрить контроль доступа, журналирование и периодический аудит.'
    ),
    'УЗ-3': (
        'СРЕДНИЙ РИСК. Ограниченный объём госидентификаторов или большой объём '
        'обычных ПДн. Рекомендуется псевдонимизация и разграничение прав доступа.'
    ),
    'УЗ-4': (
        'БАЗОВЫЙ УРОВЕНЬ. Небольшой объём обычных ПДн. '
        'Достаточно базовых организационных мер и политики конфиденциальности.'
    ),
    'Безопасно': 'ПДн не обнаружены. Специальных мер не требуется.',
}


def _get_recommendation(uz: str) -> str:
    return UZ_RECOMMENDATIONS.get(uz, '')


class PIIController:
    def __init__(self, target_dir: str, output_prefix: str = 'report') -> None:
        self.target_dir: str = target_dir
        self.output_prefix: str = output_prefix
        self.analyzer: PIIAnalyzer = PIIAnalyzer()
        self.extractor: FileExtractor = FileExtractor()

        # {путь: {stats, raw_data}}
        self.file_registry: Dict[str, Dict[str, Any]] = {}
        # {нормализованное_значение: {набор_путей}}
        self.anchor_index: Dict[str, Set[str]] = defaultdict(set)

    def _determine_uz(self, stats: Dict[str, int]) -> str:
        spec: int = int(stats.get('special_pii', 0))
        pay:  int = int(stats.get('payment_info', 0))
        gov:  int = int(stats.get('gov_ids', 0))
        com:  int = int(stats.get('common_pii', 0))

        if spec > 0:
            return 'УЗ-1'
        if pay > LARGE_VOLUME_THRESHOLD or gov > LARGE_VOLUME_THRESHOLD:
            return 'УЗ-2'
        if (0 < gov <= LARGE_VOLUME_THRESHOLD) or com > LARGE_VOLUME_THRESHOLD:
            return 'УЗ-3'
        if 0 < com <= LARGE_VOLUME_THRESHOLD:
            return 'УЗ-4'
        return 'Безопасно'

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
            text: str = str(self.extractor.extract_text(f_path))
            if not text.strip():
                continue
            res: Dict[str, Any] = self.analyzer.analyze_text(text)
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

        # Строим итоговые данные для всех трёх форматов
        report_rows: List[Dict[str, Any]] = []

        for group_id, group_files in enumerate(groups, start=1):
            # Объединяем уникальные значения по каждой категории по всей группе.
            # Один email/телефон в 2 файлах группы = 1 уникальный экземпляр.
            group_raw_union: Dict[str, Set[str]] = defaultdict(set)
            group_special_pii_total: int = 0

            for f_path in group_files:
                f_res: Dict[str, Any] = self.file_registry[f_path]
                raw_data = cast(Dict[str, List[str]], f_res.get('raw_data', {}))
                f_stats = cast(Dict[str, int], f_res.get('stats', {}))

                for cat, items in raw_data.items():
                    group_raw_union[cat].update(items)

                # special_pii — счётчик ключевых слов, не уникальных значений;
                # суммируем из каждого файла отдельно
                group_special_pii_total += int(f_stats.get('special_pii', 0))

            # Пересобираем stats из дедуплицированных множеств
            group_stats: Dict[str, int] = {
                'common_pii': (
                    len(group_raw_union['fio']) + len(group_raw_union['email']) +
                    len(group_raw_union['phone']) + len(group_raw_union['address'])
                ),
                'gov_ids': (
                    len(group_raw_union['passport_rf']) + len(group_raw_union['snils']) +
                    len(group_raw_union['inn']) + len(group_raw_union['driver_license'])
                ),
                'payment_info': (
                    len(group_raw_union['bank_card']) + len(group_raw_union['bik']) +
                    len(group_raw_union['cvv'])
                ),
                'special_pii': group_special_pii_total,
            }

            group_categories: Set[str] = {k for k, v in group_stats.items() if v > 0}
            group_counts: Dict[str, int] = {
                cat: len(vals)
                for cat, vals in group_raw_union.items() if vals
            }

            uz_level: str = self._determine_uz(group_stats)
            recommendation: str = _get_recommendation(uz_level)

            for f_path in group_files:
                row: Dict[str, Any] = {
                    'path': f_path,
                    'group_id': f'Группа_{group_id}',
                    'categories': [
                        CATEGORY_LABELS.get(c, c) for c in sorted(group_categories)
                    ],
                    'counts': {
                        CATEGORY_LABELS.get(k, k): v
                        for k, v in group_counts.items() if v > 0
                    },
                    'uz': uz_level,
                    'file_format': Path(f_path).suffix,
                    'recommendation': recommendation,
                }
                report_rows.append(row)

        # ----------------------------------------------------------------
        # Шаг 3: запись отчётов
        # ----------------------------------------------------------------
        self._write_csv(report_rows)
        self._write_json(report_rows)
        self._write_markdown(report_rows, groups)

        print(f'[+] Отчёты: {self.output_prefix}.csv / .json / .md')

    # --------------------------------------------------------------------
    # Форматы отчётов
    # --------------------------------------------------------------------

    def _write_csv(self, rows: List[Dict[str, Any]]) -> None:
        path = self.output_prefix + '.csv'
        with open(path, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([
                'Путь', 'ID_Группы', 'Категории_ПДн',
                'Количество_находок', 'УЗ', 'Формат', 'Рекомендации',
            ])
            for row in rows:
                cats = ', '.join(row['categories'])
                counts = '; '.join(
                    f"{k}: {v}" for k, v in row['counts'].items()
                )
                writer.writerow([
                    row['path'], row['group_id'], cats,
                    counts, row['uz'], row['file_format'],
                    row['recommendation'],
                ])

    def _write_json(self, rows: List[Dict[str, Any]]) -> None:
        path = self.output_prefix + '.json'
        output = {
            'generated_at': datetime.now().isoformat(),
            'total_files': len(rows),
            'results': rows,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def _write_markdown(
        self, rows: List[Dict[str, Any]], groups: List[Set[str]]
    ) -> None:
        path = self.output_prefix + '.md'

        # Статистика по УЗ
        uz_counter: Dict[str, int] = defaultdict(int)
        for row in rows:
            uz_counter[row['uz']] += 1

        lines: List[str] = [
            '# Отчёт PII-сканера',
            f'',
            f'**Дата:** {datetime.now().strftime("%d.%m.%Y %H:%M")}  ',
            f'**Файлов обработано:** {len(rows)}  ',
            f'**Групп найдено:** {len(groups)}',
            '',
            '## Сводка по уровням защищённости',
            '',
            '| УЗ | Количество файлов |',
            '|----|-------------------|',
        ]
        for uz in ['УЗ-1', 'УЗ-2', 'УЗ-3', 'УЗ-4', 'Безопасно']:
            lines.append(f'| {uz} | {uz_counter.get(uz, 0)} |')

        lines += [
            '',
            '## Детализация по файлам',
            '',
            '| Путь | Группа | Категории ПДн | УЗ | Формат |',
            '|------|--------|---------------|----|--------|',
        ]
        for row in rows:
            cats = ', '.join(row['categories']) if row['categories'] else '—'
            name = Path(row['path']).name
            lines.append(
                f'| `{name}` | {row["group_id"]} | {cats} '
                f'| **{row["uz"]}** | {row["file_format"]} |'
            )

        lines += ['', '## Рекомендации', '']
        for uz, rec in UZ_RECOMMENDATIONS.items():
            lines.append(f'### {uz}')
            lines.append(f'{rec}')
            lines.append('')

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


if __name__ == '__main__':
    app = PIIController('./test_dataset', 'report')
    app.run_scan()