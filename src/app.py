import argparse
import logging
import sys

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from router import Router
from batcher import Batch, BatchStatus
from reporter import Reporter
from scanner import Scanner
from extractor import ExtractorFactory


logging.basicConfig(
    level=logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# TODO Добавить extractor_factory из extractor и scanner из scanner
def process_batch(batch: Batch, extractor_factory: ExtractorFactory, scanner: Scanner) -> Batch:
    """Обрабатывает один батч: extraction -> scanner"""
    try:
        # Шаг 1: Извлекаем текст из файла
        extractor = ...
        batch.start_extraction()
        text = ...
        batch.finish_extraction(text)

        # Шаг 2: Находим ПДн в тексте
        batch.start_scanning()
        markup = ...
        batch.finish_scanning(markup)

    except Exception as e:
        logger.error("Батч %s завершился с ошибкой: %s", batch.id, e)
        batch.fail(e)
    return batch

def run(dataset_path: Path, output_dir: Path, workers: int):
    router = Router(dataset_path)
    batches = router.route()
    logger.info("Роутер создал %d батчей", len(batches))

    #TODO Добавить extractor_factory из extractor и scanner из scanner
    extractor_factory = ...
    scanner = ...
    reporter = Reporter(output_dir)

    # Параллельная обработка батчей
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_batch, batch, extractor_factory, scanner): batch
            for batch in batches
        }

        for future in as_completed(futures):
            batch = future.result()

            while batch.can_retry:
                logger.info("Повтор батча %s (попытка %d", batch.id, batch.attempt + 1)
                batch = process_batch(batch, extractor_factory, scanner)

            if batch.status == BatchStatus.DONE:
                reporter.add(batch.to_report())
            else:
                logger.error("Батч %s окончательно упал после %d попыток", batch.id, batch.attempt)

    # Запись отчётов
    paths = reporter.write()
    for fmt, path in paths.items():
        logger.info("Отчёт [%s]: %s", fmt, path)

def main() -> None:
    parser = argparse.ArgumentParser(description="PII Detection Pipeline")
    parser.add_argument("--dataset", required=True, help="Путь к датасету")
    parser.add_argument("--output", default='reports', help='Папка для отчётов')
    parser.add_argument("--workers", default=4, type=int, help="Количество параллельных воркеров")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Путь к датасету не существует: %s", dataset_path)
        sys.exit(1)

    run (
        dataset_path=dataset_path,
        output_dir=Path(args.output),
        workers=args.workers,
    )

if __name__ == "__main__":
    main()