import os
import csv
from pathlib import Path
from typing import Dict, Any, List, Set, cast
from collections import defaultdict, deque

from extractor import FileExtractor, ExtractionResult
from analyzer import PIIAnalyzer

LARGE_VOLUME_THRESHOLD: int = 100


class PIIController:
    def __init__(self, target_dir: str, output_csv: str) -> None:
        self.target_dir: str = target_dir
        self.output_csv: str = output_csv
        self.analyzer: PIIAnalyzer = PIIAnalyzer()
        self.extractor: FileExtractor = FileExtractor()

        self.file_registry: Dict[str, Dict[str, Any]] = {}
        self.anchor_index: Dict[str, Set[str]] = defaultdict(set)
        self.processing_statuses: Dict[str, Dict[str, str]] = {}

    def _determine_uz(self, stats: Dict[str, int]) -> str:
        """Логика определения УЗ (согласно ТЗ)."""
        spec: int = int(stats.get("special_pii", 0))
        pay: int = int(stats.get("payment_info", 0))
        gov: int = int(stats.get("gov_ids", 0))
        com: int = int(stats.get("common_pii", 0))

        if spec > 0:
            return "УЗ-1"
        if pay > LARGE_VOLUME_THRESHOLD or gov > LARGE_VOLUME_THRESHOLD:
            return "УЗ-2"
        if (0 < gov <= LARGE_VOLUME_THRESHOLD) or com > LARGE_VOLUME_THRESHOLD:
            return "УЗ-3"
        if 0 < com <= LARGE_VOLUME_THRESHOLD:
            return "УЗ-4"
        return "Безопасно"

    def _build_group_stats(self, group_files: Set[str]) -> Dict[str, Any]:
        unique_values: Dict[str, Set[str]] = {
            "fio": set(),
            "email": set(),
            "phone": set(),
            "passport_rf": set(),
            "snils": set(),
            "inn": set(),
            "bank_card": set(),
        }

        for file_path in group_files:
            file_data: Dict[str, Any] = self.file_registry[file_path]
            raw_data: Dict[str, List[str]] = cast(Dict[str, List[str]], file_data.get("raw_data", {}))
            for category, values in raw_data.items():
                if category in unique_values:
                    unique_values[category].update(values)

        group_stats: Dict[str, int] = {
            "common_pii": len(unique_values["fio"]) + len(unique_values["email"]) + len(unique_values["phone"]),
            "gov_ids": len(unique_values["passport_rf"]) + len(unique_values["snils"]) + len(unique_values["inn"]),
            "payment_info": len(unique_values["bank_card"]),
            "special_pii": 0,
        }
        group_categories: Set[str] = {category for category, values in group_stats.items() if values > 0}

        return {
            "stats": group_stats,
            "categories": sorted(group_categories),
        }

    def _find_groups(self) -> List[Set[str]]:
        """Алгоритм поиска связанных компонентов (групп файлов)."""
        visited: Set[str] = set()
        groups: List[Set[str]] = []

        for file_path in self.file_registry:
            if file_path in visited:
                continue

            current_group: Set[str] = set()
            queue = deque([file_path])
            visited.add(file_path)

            while queue:
                node: str = queue.popleft()
                current_group.add(node)

                node_data: Dict[str, Any] = self.file_registry[node]
                raw_data: Dict[str, List[str]] = cast(Dict[str, List[str]], node_data.get("raw_data", {}))

                for category in ["email", "phone", "snils", "inn", "passport_rf"]:
                    for value in raw_data.get(category, []):
                        for linked_file in self.anchor_index.get(value, set()):
                            if linked_file not in visited:
                                visited.add(linked_file)
                                queue.append(linked_file)

            groups.append(current_group)

        return groups

    def run_scan(self) -> None:
        print("[*] Шаг 1: Сканирование и сбор якорей...")
        for root, _, files in os.walk(self.target_dir):
            for file_name in files:
                file_path: str = os.path.join(root, file_name)
                extraction: ExtractionResult = self.extractor.extract_text(file_path)
                self.processing_statuses[file_path] = {
                    "status": extraction.status,
                    "error": extraction.error or "",
                }

                if extraction.status != "ok":
                    continue

                text: str = extraction.text
                if not text.strip():
                    continue

                res: Dict[str, Any] = self.analyzer.analyze_text(text)
                self.file_registry[file_path] = res

                raw_data: Dict[str, List[str]] = cast(Dict[str, List[str]], res.get("raw_data", {}))
                for category in ["email", "phone", "snils", "inn", "passport_rf"]:
                    for value in raw_data.get(category, []):
                        self.anchor_index[value].add(file_path)

        print("[*] Шаг 2: Группировка файлов и финальная классификация...")
        groups: List[Set[str]] = self._find_groups()

        with open(self.output_csv, mode="w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow([
                "Путь",
                "ID_Группы",
                "Категории_ПДн_Группы",
                "УЗ_Группы",
                "Формат",
                "Статус_Обработки",
                "Ошибка_Обработки",
            ])

            for group_id, group_files in enumerate(groups, start=1):
                group_info = self._build_group_stats(group_files)
                group_stats: Dict[str, int] = cast(Dict[str, int], group_info["stats"])
                group_categories: List[str] = cast(List[str], group_info["categories"])
                uz_level: str = self._determine_uz(group_stats)

                for file_path in group_files:
                    processing = self.processing_statuses.get(file_path, {"status": "ok", "error": ""})
                    writer.writerow([
                        file_path,
                        f"Группа_{group_id}",
                        ", ".join(group_categories),
                        uz_level,
                        Path(file_path).suffix,
                        processing["status"],
                        processing["error"],
                    ])

            grouped_files: Set[str] = set().union(*groups) if groups else set()
            for file_path, processing in self.processing_statuses.items():
                if file_path in grouped_files:
                    continue
                writer.writerow([
                    file_path,
                    "",
                    "",
                    "",
                    Path(file_path).suffix,
                    processing["status"],
                    processing["error"],
                ])

        print(f"[+] Успех! Группы сформированы и сохранены в {self.output_csv}")


if __name__ == "__main__":
    app = PIIController("./test_dataset", "report_groups.csv")
    app.run_scan()
