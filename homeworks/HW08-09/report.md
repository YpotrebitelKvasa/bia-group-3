# HW08-09 – PyTorch MLP: регуляризация и оптимизация обучения

## 1. Кратко: что сделано

Выбрал датасет EMNIST Balanced (47 классов, сложнее MNIST, но изображения 28x28 как у MNIST).  
Часть A: сравнивал регуляризацию — Baseline MLP (E1), +Dropout (E2), +BatchNorm (E3), лучший+E4 EarlyStopping.  
Часть B: диагностика LR (O1/O2) + SGD+momentum+weight_decay (O3).

## 2. Среда и воспроизводимость

- Python: 3.12.8
- torch / torchvision: PyTorch: 2.1.0+cu121
- Устройство (CPU/GPU): CUDA (NVIDIA GeForce RTX 2050 Laptop GPU)
- Seed: 42
- Как запустить: открыть `HW08-09.ipynb` и выполнить Run All.

## 3. Данные

- Датасет: (KMNIST / EMNIST Balanced / CIFAR10) EMNIST Balanced
- Разделение: train/val/test (например, train split 80/20 + test из torchvision) train 80/20 val + test из torchvision (fast_mode=12000/3000) 
- Трансформации (transform): (ToTensor / Normalize / другое) ToTensor + Normalize(0.5, 0.5)
- Комментарий (1-3 предложения): 47 классов (буквы+цифры), изображения 28x28, сложность выше MNIST из-за большего числа классов.

## 4. Базовая модель и обучение

- Модель MLP (кратко): 3 скрытых слоя [512,256,128], ReLU
- Loss: CrossEntropyLoss
- Базовый Optimizer (для части A): Adam lr=1e-3
- Batch size: 128
- Epochs (макс): 20 (E1-E3),  до 50 (E4) 
- EarlyStopping: patience=4, metric=val_accuracy

## 5. Часть A (S08): регуляризация (E1-E4)

Опишите, что меняли. Формулировки должны быть короткими и сопоставимыми.

- E1 (base): [512,256,128] без Dropout/BatchNorm
- E2 (Dropout): как E1 + Dropout(p=0.3) 
- E3 (BatchNorm): как E1 + BatchNorm 
- E4 (EarlyStopping): E3 (лучший) + EarlyStopping(patience=4)

## 6. Часть B (S09): LR, оптимизаторы, weight decay (O1-O3)

- O1: LR слишком большой (Adam lr=0.1)
- O2: LR слишком маленький (Adam lr=1e-5)
- O3: SGD+momentum (momentum=0.9) + weight_decay=1e-4= (lr=1e-2)

## 7. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель: `./artifacts/best_model.pt`
- Конфиг лучшей модели: `./artifacts/best_config.json`
- Кривые лучшего прогона: `./artifacts/figures/curves_best.png`
- Кривые “плохих LR”: `./artifacts/figures/curves_lr_extremes.png` и `./artifacts/figures/curves_lr_extremes2.png'

Короткая сводка (5-9 строк):

- Лучший эксперимент части A: (E2/E3/E4) E4 
- Лучшая val_accuracy: 0.78 (E3)
- Итоговая test_accuracy (для лучшей модели): 0.7643
- Что видно на O1 (слишком большой LR): Нестабильное обучение, loss не снижается плавно, accuracy растет медленно и остается низкой.
- Что видно на O2 (слишком маленький LR): Очень медленное обучение, loss снижается крайне медленно, accuracy не успевает вырасти за 6 эпох.
- Как повёл себя O3 (SGD+momentum + weight decay) относительно Adam (по кривым/метрике): стабильно, но хуже Adam

## 8. Анализ

(8-15 предложений)

- Где на графиках видно переобучение (если видно) и как его изменили Dropout/BatchNorm. На графиках E1 (baseline) видно классическое переобучение: train loss падает до конца, а val loss стабилизируется примерно после 11 эпохи (val_acc=0.762). E2 Dropout(0.3) немного сгладил расхождение (val_acc=0.773), но эффект слабый. E3 BatchNorm полностью устранил переобучение — val/train loss идут параллельно, достигнув 0.780 (лучший результат).
- Что сделал EarlyStopping (на какой эпохе остановил, помог ли). E4 на базе E3 остановился на 15-й эпохе (patience=4, max_epochs=20). Это предотвратило бесполезные 5 эпох без улучшения, сохранив test_accuracy=0.7643. Без EarlyStopping модель бы переобучилась на последних эпохах.
- Почему O1 и O2 “плохие” (что конкретно видно по loss/accuracy). O1 — слишком большой LR (lr=0.1).
Градиентный взрыв - loss взлетает до 3.86, val_acc падает до 0.025 за 6 эпох. Оптимизатор "перепрыгивает" минимум, градиенты расходятся O2 — слишком маленький LR (lr=1e-5). Полный застой - loss едва сдвинулся с 3.61, val_acc 0.164 за 6 эпох. Шаг слишком мал — сеть "ползёт" к минимуму.
- Что даёт SGD+momentum и зачем добавляют weight decay (что заметили в ваших результатах). SGD (lr=0.01, momentum=0.9, wd=1e-4) достиг 0.701 — стабильно, но на 10% хуже Adam. Momentum ускоряет SGD в плоских областях, weight_decay=L2 предотвращает переобучение (val/train loss близки).
- Почему выбранный лучший конфиг разумен именно для этого датасета. 47 классов создают сложное многомерное распределение. BatchNorm нормализует внутренние представления, стабилизирует градиенты и ускоряет сходимость на большом числе классов. Для EMNIST это эффективнее Dropout (0.780 vs 0.773)

## 9. Итоговый вывод

(3-7 предложений)

Лучший конфиг: MLP[512,256,128] + BatchNorm + EarlyStopping(patience=4) + Adam lr=1e-3. 0.780 val_acc — лучший результат. 
Улучшения:  
1. Learning Rate Scheduler после плато.  
2. RandomRotation(±10°) для аугментации.

## 10. Приложение (опционально)

Если вы делали дополнительные сравнения:

- чистое сравнение Adam vs SGD на одном lr без weight decay
- другая активация / инициализация
- дополнительные графики: `./artifacts/figures/...`
