import logging
from pathlib import Path

from batcher import Batch

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {
    # Документы
    "pdf", "doc", "docx", "rtf", "html",
    # Таблицы
    "csv", "xls", "xlsx", "parquet",
    # Данные
    "json",
    # Изображения
    "png", "jpg", "jpeg", "gif", "bmp",
    # Медиа
    "mp4",
}

class Router:
    """
    Рекурсивно обходит директорию датасета, фильтрует файлы по формату и создаёт батчи
    """

    def __init__(self, dataset_path: str | Path, formats: set[str] | None = None, run_id: str |None = None,):
        self.dataset_path = Path(dataset_path)
        self.formats = formats or SUPPORTED_FORMATS
        self.run_id = run_id
        self._seen_hashes: set[str] = set()

    def route(self) -> list[Batch]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Датасет не найден: {self.dataset_path}")

        batches = []

        for file_path in self._iter_files():
            batch = self._make_batch(file_path)

            if batch is None:
                logger.warning("Файл пропущен (дубликат): %s", file_path.name)
                continue

            batches.append(batch)
            logger.info("Батч создан: %s [%s]", file_path.name, batch.file_format)

        logger.info(
            "Роутер завершил обход: %d батчей из %s",
            len(batches), self.dataset_path,
        )
        return batches

    def _iter_files(self):
        """Рекурсивно перебрать все файлы датасета."""
        for file_path in sorted(self.dataset_path.rglob("*")):
            if not file_path.is_file():
                continue
            ext = file_path.suffix.lower().lstrip(".")
            if ext not in self.formats:
                logger.debug("Пропущен неподдерживаемый формат: %s", file_path.name)
                continue
            yield file_path

    def _make_batch(self, file_path: Path) -> Batch | None:
        """Создать батч, пропустить если файл уже обрабатывался."""
        batch = Batch(
            file_path=file_path,
            metadata={"run_id": self.run_id},
        )

        # Дедупликация по хешу
        file_hash = self._hash(file_path)
        if file_hash in self._seen_hashes:
            logger.warning("Дубликат пропущен: %s", file_path.name)
            return None
        self._seen_hashes.add(file_hash)

        batch.metadata["file_hash"] = file_hash
        return batch

    @staticmethod
    def _hash(file_path: Path) -> str:
        """SHA-256 хеш файла для дедупликации."""
        import hashlib
        sha = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(65_536), b""):
                sha.update(chunk)
        return sha.hexdigest()