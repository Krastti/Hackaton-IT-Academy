# Используем Python 3.11 slim-версию для оптимального размера образа
FROM python:3.11-slim

# Устанавливаем системные зависимости, необходимые для OpenCV и EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Задаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt .

# Устанавливаем зависимости Python.
# ML-пакеты (torch, easyocr, natasha) довольно объемные, поэтому отключаем кэш,
# чтобы не раздувать итоговый размер образа.
RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Создаем структуру проекта (в app.py используются импорты вида `from src.*`)
# и папки для монтирования данных
RUN mkdir -p src data reports

# Копируем точку входа
COPY app.py .

# Копируем модули в папку src/
COPY __init__.py src/
COPY batcher.py src/
COPY extractor.py src/
COPY reporter.py src/
COPY router.py src/
COPY scanner.py src/

# Задаем точку входа
ENTRYPOINT ["python", "app.py"]

# Аргументы по умолчанию (покажет справку, если запустить без параметров)
CMD ["--help"]