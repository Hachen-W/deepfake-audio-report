#!/bin/bash
# Запуск прототипа. Сам создаёт окружение и доставляет зависимости.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "Создаю виртуальное окружение..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Переустанавливаем зависимости только если requirements.txt изменился
STAMP=".venv/.requirements.sha"
CURRENT=$(sha256sum requirements.txt | cut -d' ' -f1)
if [ "$(cat "$STAMP" 2>/dev/null)" != "$CURRENT" ]; then
    echo "Устанавливаю зависимости..."
    python -m pip install -q --upgrade pip
    python -m pip install -q -r requirements.txt
    echo "$CURRENT" > "$STAMP"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg в системе не найден: M4A/WMA/AMR читаются через imageio-ffmpeg,"
    echo "но сведения о кодеке в отчёт не попадут (нет ffprobe)."
    echo "Полная поддержка: sudo apt install ffmpeg"
fi

export QT_LOGGING_RULES="qt.accessibility.atspi=false"
exec env -u LD_LIBRARY_PATH -u GTK_PATH -u QT_PLUGIN_PATH python app.py
