FROM apache/airflow:2.8.1-python3.11

USER root
# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Настройка стабильного зеркала (Tsinghua University) — оно содержит все версии и работает лучше в РФ
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Обновляем pip и устанавливаем поддержку SOCKS
RUN pip install --no-cache-dir --upgrade pip "httpx[socks]" aiohttp-socks

# Установка зависимостей из requirements.txt
COPY --chown=airflow:root requirements.txt .
RUN pip install --no-cache-dir --timeout 1000 --retries 10 -r requirements.txt

# Отдельная установка Torch CPU (экономия места)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Установка Mystem (бинарный файл)
USER root
RUN apt-get update && apt-get install -y wget && \
    wget http://download.cdn.yandex.net/mystem/mystem-3.1-linux-64bit.tar.gz && \
    tar -xvf mystem-3.1-linux-64bit.tar.gz && \
    mv mystem /usr/local/bin/mystem && \
    rm mystem-3.1-linux-64bit.tar.gz && \
    chmod +x /usr/local/bin/mystem

USER airflow

# Копируем наш код в контейнер
COPY --chown=airflow:root src /opt/airflow/src
COPY --chown=airflow:root .env.db /opt/airflow/.env.db
COPY --chown=airflow:root .env.scraper /opt/airflow/.env.scraper

# Устанавливаем PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow"