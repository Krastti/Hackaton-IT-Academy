import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

class BatchStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    SCANNING = "scanning"
    DONE = "done"
    FAILED = "failed"

@dataclass
class Batch:
    file_path: Path
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: BatchStatus = BatchStatus.PENDING

    # Метаданные файлы
    file_format: str = ""
    metadata: dict = field(default_factory=dict)

    # Данные пайплайна
    extracted_text: str = ""
    scan_markup: dict = field(default_factory=dict)

    # Retry
    attempt: int = 0
    max_retries: int = 3
    errors: list = field(default_factory=list)

    def __post_init__(self):
        self.file_path = Path(self.file_path)
        self.file_format = self.file_path.suffix.lower().lstrip(".")

    # Экстрактор вызывает эти методы
    def start_extraction(self):
        self.attempt += 1
        self.status = BatchStatus.EXTRACTING

    def finish_extraction(self, text: str):
        self.extracted_text = text
        self.status = BatchStatus.SCANNING

    # Сканер вызывает эти методы
    def start_scanning(self):
        self.status = BatchStatus.SCANNING

    def finish_scanning(self, markup: dict):
        self.scan_markup = markup
        self.status = BatchStatus.DONE

    # Обработка ошибок
    def fail(self, error: Exception):
        self.errors.append({
            "attempt": self.attempt,
            "error": str(error),
        })
        self.status = BatchStatus.FAILED

    @property
    def can_retry(self) -> bool:
        return self.status == BatchStatus.FAILED and self.attempt < self.max_retries

    def to_report(self) -> dict:
        return {
            "batch_id": self.id,
            "file_path": str(self.file_path),  # 1. Путь
            "file_format": self.file_format,  # 5. Формат файла
            "categories": self.scan_markup.get("categories", []),  # 2. Список категорий
            "total_matches": len(self.scan_markup.get("matches", [])),  # 3. Кол-во экземпляров
            "protection_level": self.scan_markup.get("protection_level"),  # 4. Уровень защиты
            "recommendations": self.scan_markup.get("recommendations", []),  # 6. Рекомендации
            "metadata": self.metadata,
        }