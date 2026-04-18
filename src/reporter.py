import csv
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class Reporter:
    """
    Принимает отчёты от батчей и записывает итоговые файлы.

    Использование:
        reporter = Reporter(output_dir="reports/")
        reporter.add(batch.to_report())
        reporter.add(batch.to_report())
        reporter.write()
    """
    def __init__(self, output_dir: str | Path = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._reports: list[dict] = []

    def add(self, report: dict) -> None:
        """Добавляет отчёт одного батча"""
        self._reports.append(report)
        logger.info("Reporter: Добавлен репорт для %s", report.get("file_path"))

    def write(self) -> dict[str, Path]:
        """
        Записывает все накопленные отчёты на диск
        :return: возвращает пути к созданным файлам
        """
        if not self._reports:
            logger.warning("Reporter: Нет репортов для записи")
            return {}

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        paths = {"csv": self._write_csv(timestamp)}

        logger.info("Reporter: %d репортов в %s", len(self._reports), self.output_dir)
        return paths

    def _write_csv(self, timestamp: str) -> Path:
        path = self.output_dir / f"reports_{timestamp}.csv"
        fieldnames = [
            "batch_id",
            "file_path",  # 1. Путь
            "categories",  # 2. Список категорий
            "total_matches",  # 3. Кол-во экземпляров
            "protection_level",  # 4. Уровень защиты
            "file_format",  # 5. Формат файла
            "recommendations",  # 6. Рекомендации
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in self._reports:
                writer.writerow({
                    "batch_id":         r.get("batch_id"),
                    "file_path":        r.get("file_path"),
                    "categories":       ", ".join(r.get("categories", [])),
                    "total_matches":    r.get("total_matches", 0),
                    "protection_level": r.get("protection_level"),
                    "file_format":      r.get("file_format"),
                    "recommendations":  "; ".join(r.get("recommendations", [])),
                })
        logger.info("Reporter: CSV файл записан в %s", path)
        return path