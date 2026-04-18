import argparse
import logging
import sys
import threading

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

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
worker_progress_lock = threading.Lock()
worker_progress_bars: dict[str, tqdm] = {}

def process_batch(batch: Batch, extractor_factory: ExtractorFactory, scanner: Scanner) -> Batch:
    """Обрабатывает один батч: extraction -> scanner"""
    try:
        batch.metadata["worker_name"] = threading.current_thread().name
        logger.info(
            "Воркер %s обрабатывает файл: %s",
            threading.current_thread().name,
            batch.file_path,
        )
        # Шаг 1: Извлекаем текст из файла
        extractor = extractor_factory.get(batch.file_format)
        batch.start_extraction()
        text = extractor.extract(batch.file_path)
        batch.finish_extraction(text)

        # Шаг 2: Находим ПДн в тексте
        batch.start_scanning()
        markup = scanner.scan(batch.extracted_text)
        batch.finish_scanning(markup)

    except Exception as e:
        logger.error("Батч %s завершился с ошибкой: %s", batch.id, e)
        batch.fail(e)
    return batch

def _get_worker_progress_bar(worker_name: str, total: int) -> tqdm:
    with worker_progress_lock:
        bar = worker_progress_bars.get(worker_name)
        if bar is None:
            bar = tqdm(
                total=total,
                desc=worker_name,
                unit="file",
                dynamic_ncols=True,
            )
            worker_progress_bars[worker_name] = bar
        return bar

def run(dataset_path: Path, output_dir: Path, workers: int):
    router = Router(dataset_path)
    batches = router.route()
    logger.info("Роутер создал %d батчей", len(batches))

    extractor_factory = ExtractorFactory()
    scanner = Scanner()
    reporter = Reporter(output_dir)

    # Параллельная обработка батчей
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_batch, batch, extractor_factory, scanner): batch
            for batch in batches
        }

        for future in as_completed(futures):
            batch = future.result()
            worker_name = batch.metadata.get("worker_name", threading.current_thread().name)
            bar = _get_worker_progress_bar(worker_name, len(batches))
            bar.update(1)

            while batch.can_retry:
                logger.info("Повтор батча %s (попытка %d)", batch.id, batch.attempt + 1)
                batch = pool.submit(
                    process_batch,
                    batch,
                    extractor_factory,
                    scanner,
                ).result()
                worker_name = batch.metadata.get("worker_name", threading.current_thread().name)
                bar = _get_worker_progress_bar(worker_name, len(batches))
                bar.update(1)

            if batch.status == BatchStatus.DONE:
                reporter.add(batch.to_report())
            else:
                logger.error("Батч %s окончательно упал после %d попыток", batch.id, batch.attempt)

    # Запись отчётов
    paths = reporter.write()
    for fmt, path in paths.items():
        logger.info("Отчёт [%s]: %s", fmt, path)

    for bar in worker_progress_bars.values():
        bar.close()
    worker_progress_bars.clear()

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