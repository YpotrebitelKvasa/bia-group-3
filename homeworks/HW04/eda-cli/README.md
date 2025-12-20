# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

Количество числовых столбцов для гистограмм (по умолчанию 6):
```bash
uv run eda-cli report data/example.csv --out-dir reports
```

### Новые параметры команды report

#### --max-hist-columns
Количество числовых столбцов для гистограмм (по умолчанию 6):
```bash
uv run eda-cli report data/example.csv --max-hist-columns 10
```

#### --top-k-categories
Количество top-значений для категориальных признаков (по умолчанию 5):
```bash
uv run eda-cli report data/example.csv --top-k-categories 10
```

#### --min-missing-share
Порог доли пропусков для "проблемных" колонок (по умолчанию 0.3):
```bash
uv run eda-cli report data/example.csv --min-missing-share 0.5
```

### Пример: полный анализ с кастомными параметрами
```bash
uv run eda-cli report data/example.csv --max-hist-columns 8 --top-k-categories 10 --min-missing-share 0.2 --out-dir reports/q4_analysis
```
###


В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```
