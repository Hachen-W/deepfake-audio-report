#!/bin/bash
# Запуск прототипа. Сам создаёт окружение и доставляет зависимости.
# Ключ --reinstall принудительно переставляет пакеты.
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "--reinstall" ]; then
    rm -f .venv/.requirements.sha
    shift
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "Не найден python3." >&2
    exit 1
fi

if [ ! -f requirements.txt ] || [ ! -f app.py ]; then
    echo "Запускать нужно из каталога проекта: нет requirements.txt или app.py." >&2
    exit 1
fi

# Проверяем сам интерпретатор, а не каталог: битое окружение пересоздаём
if [ ! -x .venv/bin/python ]; then
    echo "Создаю виртуальное окружение..."
    rm -rf .venv
    if ! python3 -m venv .venv; then
        echo "Не удалось создать окружение. Поставьте: sudo apt install python3-venv" >&2
        exit 1
    fi
fi
source .venv/bin/activate

# Переустанавливаем зависимости только если requirements.txt изменился
STAMP=".venv/.requirements.sha"
CURRENT=$(sha256sum requirements.txt | cut -d' ' -f1)
if [ "$(cat "$STAMP" 2>/dev/null)" != "$CURRENT" ]; then
    echo "Устанавливаю зависимости..."
    python -m pip install -q --upgrade pip
    if ! python -m pip install -r requirements.txt; then
        echo "Зависимости не установились. Повторите после исправления." >&2
        exit 1
    fi
    echo "$CURRENT" > "$STAMP"
fi

# --- проверки системного окружения: не блокируют запуск, только предупреждают ---

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg не найден: часть форматов и извлечение звука из видео"
    echo "будут работать через imageio-ffmpeg. Полная поддержка: sudo apt install ffmpeg"
elif ! command -v ffprobe >/dev/null 2>&1; then
    echo "ffprobe не найден: в отчёт не попадут сведения о контейнере и кодеке."
fi

# Список библиотек забираем один раз: grep -q в конвейере обрывает ldconfig,
# и при set -o pipefail это выглядит как «библиотека не найдена»
SYSTEM_LIBS=$(ldconfig -p 2>/dev/null || true)

case "$SYSTEM_LIBS" in
    *libxcb-cursor*) ;;
    *)
        echo "Нет libxcb-cursor0 — Qt может не загрузить плагин xcb."
        echo "Поставьте: sudo apt install libxcb-cursor0"
        ;;
esac

case "$SYSTEM_LIBS" in
    *libpulse.so*|*libasound.so*) ;;
    *)
        echo "Не видно звукового вывода — кнопка «Слушать» работать не будет."
        echo "Поставьте: sudo apt install libpulse0 gstreamer1.0-pulseaudio"
        ;;
esac

export QT_LOGGING_RULES="qt.accessibility.atspi=false"
export MPLBACKEND=QtAgg

# Чистим переменные snap-окружения: из-под VS Code они ломают загрузку плагинов Qt
exec env -u LD_LIBRARY_PATH -u GTK_PATH -u QT_PLUGIN_PATH python app.py "$@"
