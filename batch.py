"""Пакетная обработка папки: прогон детектора по всем фонограммам."""
import csv
import os

import analysis

EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac",
              ".wma", ".aiff", ".amr", ".mp4", ".mkv", ".avi", ".mov"}

COLUMNS = ["файл", "длительность", "оценок", "максимум", "средняя",
           "участков", "вердикт", "ошибка"]


def find_files(folder):
    """Все подходящие файлы в папке, включая вложенные."""
    found = []
    for root, _, names in os.walk(folder):
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() in EXTENSIONS:
                found.append(os.path.join(root, name))
    return sorted(found)


def process_file(path, detector, threshold):
    """Одна запись. Ошибки не прерывают пакет, а попадают в таблицу."""
    row = {name: "" for name in COLUMNS}
    row["файл"] = path
    temp_path = None
    try:
        work_path = path
        if analysis.is_video(path):
            temp_path = analysis.extract_audio(path, 0)
            work_path = temp_path

        x, sr = analysis.load_audio(work_path)
        row["длительность"] = round(len(x) / sr, 2)

        times, probs = detector.scores(x, sr, work_path)
        row["оценок"] = len(probs)
        if len(probs):
            row["максимум"] = round(float(probs.max()), 4)
            row["средняя"] = round(float(probs.mean()), 4)
            row["участков"] = len(
                analysis.intervals_above(times, probs, threshold))

        result = detector.last_result or {}
        row["вердикт"] = result.get("verdict", "")
    except Exception as e:
        row["ошибка"] = str(e)[:200]
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    return row


def run_folder(folder, detector, threshold, on_progress=None):
    """Прогоняет детектор по всей папке, возвращает список строк."""
    files = find_files(folder)
    rows = []
    for number, path in enumerate(files, start=1):
        if on_progress:
            on_progress(f"[{number}/{len(files)}] {os.path.basename(path)}")
        detector.last_result = None
        rows.append(process_file(path, detector, threshold))
    return rows


def save_csv(rows, out_path):
    """Сводная таблица. utf-8-sig — чтобы Excel не ломал кириллицу."""
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    return out_path
