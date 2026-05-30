"""
Конфигурация pytest: добавляет корень проекта в sys.path,
чтобы тесты могли импортировать модули из src/
"""
import sys
from pathlib import Path

# Добавляем корень проекта (родительская папка от tests/) в начало sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))