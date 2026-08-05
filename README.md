"""Формирование отчёта в DOCX."""
import hashlib
import os
from datetime import datetime

from docx import Document
from docx.shared import Inches


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_report(out_path, wav_path, sr, duration, checks, intervals,
                 threshold, model_name, plot_png=None, info=None):
    doc = Document()
    doc.add_heading("Заключение по исследованию фонограммы", level=0)
    doc.add_paragraph(f"Дата формирования: {datetime.now():%d.%m.%Y %H:%M}")

    doc.add_heading("1. Объект исследования", level=1)
    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    rows = [
        ("Имя файла", os.path.basename(wav_path)),
        ("Размер, байт", str(os.path.getsize(wav_path))),
        ("SHA-256", file_sha256(wav_path)),
        ("Контейнер / кодек", f"{(info or {}).get('container', '?')} / {(info or {}).get('codec', '?')}"),
        ("Частота дискретизации, Гц", str(sr)),
        ("Длительность, с", f"{duration:.2f}"),
    ]
    for name, value in rows:
        cells = table.add_row().cells
        cells[0].text = name
        cells[1].text = value

    doc.add_heading("2. Методика", level=1)
    doc.add_paragraph(
        f"Модель детектирования: {model_name}. "
        f"Окно анализа 2.0 с, шаг 0.5 с, порог принятия решения {threshold}."
    )

    doc.add_heading("3. Признаки синтезированной речи", level=1)
    if plot_png:
        doc.add_picture(plot_png, width=Inches(6))
    if intervals:
        doc.add_paragraph("Участки с оценкой выше порога:")
        for t0, t1 in intervals:
            doc.add_paragraph(f"{t0:.2f} — {t1:.2f} с", style="List Bullet")
    else:
        doc.add_paragraph("Участков с оценкой выше порога не выявлено.")

    doc.add_heading("4. Классические признаки обработки", level=1)
    doc.add_paragraph(f"Смещение постоянной составляющей: {checks['dc_offset']:.6f}")
    doc.add_paragraph(f"Частота среза спектра: {checks['cutoff_hz']:.0f} Гц")
    doc.add_paragraph(f"Участков с постоянным значением отсчётов: {len(checks['constant_runs'])}")
    doc.add_paragraph(f"Пар повторяющихся фрагментов: {len(checks['repeats'])}")
    if (info or {}).get("lossy"):
        doc.add_paragraph(
            "Фонограмма представлена в формате со сжатием с потерями. Частота среза "
            "спектра в этом случае объясняется работой кодека и не свидетельствует "
            "о редактировании. Исследование выполнено по декодированному сигналу."
        )

    doc.add_heading("5. Ограничения", level=1)
    doc.add_paragraph(
        "Результат носит вероятностный характер и не является выводом о подлинности "
        "записи. Прототип не прошёл валидацию по методике экспертного учреждения; "
        "оценки на фонограммах, полученных неизвестными системами синтеза, "
        "могут быть занижены."
    )

    doc.save(out_path)
    return out_path
  