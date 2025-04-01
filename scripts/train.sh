#!/bin/bash

# Активация виртуального окружения (если используется)
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Запуск основного скрипта
python main.py