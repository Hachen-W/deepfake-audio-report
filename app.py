"""Демо-прототип: анализ фонограммы на признаки синтезированной речи."""
import os
import sys

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                               QMainWindow, QPushButton, QTextEdit,
                               QVBoxLayout, QWidget)

import analysis
import report

MODEL_PATH = "model.onnx"
THRESHOLD = 0.7


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ фонограммы — прототип")
        self.resize(1000, 700)

        self.wav_path = None
        self.signal = None
        self.sr = None
        self.times = None
        self.probs = None
        self.checks = None
        self.info = {}
        self.session = self.load_model()

        self.figure = Figure(figsize=(9, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumHeight(160)
        self.status = QLabel(f"Модель: {self.model_name}")

        open_btn = QPushButton("Открыть аудио")
        open_btn.clicked.connect(self.open_file)
        run_btn = QPushButton("Анализ")
        run_btn.clicked.connect(self.run_analysis)
        report_btn = QPushButton("Отчёт DOCX")
        report_btn.clicked.connect(self.save_report)

        buttons = QHBoxLayout()
        for widget in (open_btn, run_btn, report_btn, self.status):
            buttons.addWidget(widget)
        buttons.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(buttons)
        layout.addWidget(self.canvas)
        layout.addWidget(self.text)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def load_model(self):
        """Пробует загрузить ONNX-модель, иначе работает на заглушке."""
        if os.path.exists(MODEL_PATH):
            import onnxruntime
            self.model_name = MODEL_PATH
            return onnxruntime.InferenceSession(MODEL_PATH)
        self.model_name = "ЗАГЛУШКА (модель не подключена)"
        return None

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", "",
            "Аудио (*.wav *.mp3 *.flac *.ogg *.opus *.m4a *.aac *.wma *.aiff *.amr);;Все файлы (*)")
        if not path:
            return
        self.wav_path = path
        self.signal, self.sr = analysis.load_audio(path)
        self.info = analysis.probe_format(path)
        note = " — сжатие с потерями" if self.info.get("lossy") else ""
        self.text.setPlainText(
            f"Загружен {os.path.basename(path)}\n"
            f"{self.info.get('codec', '?')}{note}, {self.sr} Гц, "
            f"{len(self.signal) / self.sr:.2f} с"
        )
        self.draw()

    def run_analysis(self):
        if self.signal is None:
            return
        self.times, self.probs = analysis.frame_probabilities(
            self.signal, self.sr, self.session)
        self.checks = analysis.classic_checks(self.signal, self.sr)
        intervals = analysis.intervals_above(self.times, self.probs, THRESHOLD)

        lines = [f"Модель: {self.model_name}", ""]
        if intervals:
            lines.append("Участки выше порога:")
            lines += [f"  {t0:.2f} — {t1:.2f} с" for t0, t1 in intervals]
        else:
            lines.append("Участков выше порога не выявлено.")
        lines += [
            "",
            f"Смещение постоянной составляющей: {self.checks['dc_offset']:.6f}",
            f"Частота среза спектра: {self.checks['cutoff_hz']:.0f} Гц",
            f"Постоянных участков: {len(self.checks['constant_runs'])}",
            f"Повторяющихся пар: {len(self.checks['repeats'])}",
        ]
        self.text.setPlainText("\n".join(lines))
        self.draw()

    def draw(self):
        self.figure.clear()
        top = self.figure.add_subplot(211)
        top.specgram(self.signal, Fs=self.sr, cmap="magma")
        top.set_ylabel("Частота, Гц")

        bottom = self.figure.add_subplot(212, sharex=top)
        if self.probs is not None:
            bottom.plot(self.times, self.probs)
            bottom.axhline(THRESHOLD, linestyle="--", color="red")
            for t0, t1 in analysis.intervals_above(self.times, self.probs, THRESHOLD):
                bottom.axvspan(t0, t1, alpha=0.3, color="red")
        bottom.set_ylim(0, 1)
        bottom.set_xlabel("Время, с")
        bottom.set_ylabel("Оценка")

        self.figure.tight_layout()
        self.canvas.draw()

    def save_report(self):
        if self.checks is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчёт", "report.docx",
                                              "DOCX (*.docx)")
        if not path:
            return
        png = "_plot.png"
        self.figure.savefig(png, dpi=120)
        intervals = analysis.intervals_above(self.times, self.probs, THRESHOLD)
        report.build_report(path, self.wav_path, self.sr,
                            len(self.signal) / self.sr, self.checks, intervals,
                            THRESHOLD, self.model_name, png, self.info)
        self.text.append(f"\nОтчёт сохранён: {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    