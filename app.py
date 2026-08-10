"""Демо-прототип: анализ фонограммы на признаки синтезированной речи."""
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox,
                               QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                               QMainWindow, QPushButton, QTextEdit,
                               QVBoxLayout, QWidget)

import analysis
import batch
import config
import detectors
import player
import report

FILE_FILTER = (
    "Аудио и видео (*.wav *.mp3 *.flac *.ogg *.opus *.m4a *.aac *.wma "
    "*.aiff *.amr *.mp4 *.mkv *.avi *.mov);;Все файлы (*)")


class AnalysisTask(QThread):
    """Считает кривую выбранным детектором в отдельном потоке."""
    done = Signal(object, object)
    failed = Signal(str)

    def __init__(self, detector, signal, sr, path):
        super().__init__()
        self.detector = detector
        self.signal = signal
        self.sr = sr
        self.path = path

    def run(self):
        try:
            times, probs = self.detector.scores(self.signal, self.sr, self.path)
            self.done.emit(times, probs)
        except Exception as e:
            self.failed.emit(str(e))


class BatchTask(QThread):
    """Пакетная обработка папки в отдельном потоке."""
    progress = Signal(str)
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, folder, detector, threshold):
        super().__init__()
        self.folder = folder
        self.detector = detector
        self.threshold = threshold

    def run(self):
        try:
            rows = batch.run_folder(self.folder, self.detector, self.threshold,
                                    self.progress.emit)
            self.done.emit(rows)
        except Exception as e:
            self.failed.emit(str(e))


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
        self.service_result = None
        self.task = None
        self.results = {}       # результаты всех запущенных детекторов
        self.journal = []       # журнал сеанса для отчёта
        self.source_path = None  # исходный файл, если звук извлечён из видео
        self.temp_path = None    # извлечённая дорожка, удаляется при смене файла

        self.player = player.Player()
        self.cursors = []        # вертикальные линии позиции на обоих графиках
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(100)
        self.play_timer.timeout.connect(self.update_cursor)

        self.detectors = detectors.available()
        self.detector_box = QComboBox()
        for detector in self.detectors:
            self.detector_box.addItem(detector.name)
        self.detector_box.currentIndexChanged.connect(self.rebuild_params)

        # Порог общий для всех детекторов: по нему выделяются участки
        self.threshold_box = QDoubleSpinBox()
        self.threshold_box.setRange(0.0, 1.0)
        self.threshold_box.setSingleStep(0.05)
        self.threshold_box.setDecimals(2)
        self.threshold_box.setValue(config.THRESHOLD)
        self.threshold_box.valueChanged.connect(lambda _: self.draw())

        # Вторая строка: всё, что калибруется. Порог постоянный,
        # поля детектора перестраиваются при его смене.
        self.param_row = QHBoxLayout()
        self.param_widgets = {}

        self.numbers_row = QHBoxLayout()
        self.numbers_row.addWidget(QLabel("Порог"))
        self.numbers_row.addWidget(self.threshold_box)
        self.numbers_row.addLayout(self.param_row)

        self.figure = Figure(figsize=(9, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumHeight(160)
        self.status = QLabel("Файл не выбран")

        open_btn = QPushButton("Открыть файл")
        open_btn.clicked.connect(self.open_file)
        run_btn = QPushButton("Анализ")
        run_btn.clicked.connect(self.run_analysis)
        self.play_btn = QPushButton("Слушать")
        self.play_btn.clicked.connect(self.toggle_play)
        batch_btn = QPushButton("Папка")
        batch_btn.clicked.connect(self.run_batch)
        report_btn = QPushButton("Отчёт DOCX")
        report_btn.clicked.connect(self.save_report)

        buttons = QHBoxLayout()
        for widget in (open_btn, self.play_btn, self.detector_box, run_btn,
                       batch_btn, report_btn, self.status):
            buttons.addWidget(widget)
        buttons.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(buttons)          # первая строка — кнопки
        layout.addLayout(self.numbers_row)  # вторая строка — что калибруем
        layout.addWidget(self.canvas)
        layout.addWidget(self.text)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.rebuild_params()

    def toggle_play(self):
        """Запускает или останавливает прослушивание."""
        if self.signal is None:
            return
        if self.play_timer.isActive():
            self.stop_play()
            return
        try:
            self.player.play(self.signal, self.sr)
        except Exception as e:
            self.log(f"Не удалось воспроизвести: {e}")
            return
        self.play_btn.setText("Стоп")
        self.play_timer.start()

    def stop_play(self):
        self.play_timer.stop()
        self.player.stop()
        self.play_btn.setText("Слушать")
        self.move_cursor(None)

    def update_cursor(self):
        """Двигает отметку позиции по обоим графикам."""
        if not self.player.is_playing():
            self.stop_play()
            return
        self.move_cursor(self.player.position())

    def move_cursor(self, position):
        """position в секундах, None — спрятать отметку."""
        for line in self.cursors:
            if position is None:
                line.set_visible(False)
            else:
                line.set_visible(True)
                line.set_xdata([position, position])
        self.canvas.draw_idle()

    def cleanup_temp(self):
        """Удаляет дорожку, извлечённую из предыдущего видео."""
        if self.temp_path and os.path.exists(self.temp_path):
            os.remove(self.temp_path)
        self.temp_path = None

    def closeEvent(self, event):
        self.stop_play()
        self.cleanup_temp()
        super().closeEvent(event)

    def run_batch(self):
        """Прогоняет выбранный детектор по всем файлам папки."""
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if not folder:
            return
        detector = self.current_detector()
        detector.values = {k: w.value() for k, w in self.param_widgets.items()}
        count = len(batch.find_files(folder))
        if count == 0:
            self.log("В папке нет подходящих файлов")
            return

        self.log(f"Пакетная обработка: {count} файлов, детектор {detector.name}")
        self.task = BatchTask(folder, detector, self.threshold())
        self.task.progress.connect(self.log)
        self.task.done.connect(self.on_batch_done)
        self.task.failed.connect(lambda msg: self.log(f"Ошибка пакета: {msg}"))
        self.task.start()

    def on_batch_done(self, rows):
        errors = sum(1 for row in rows if row["ошибка"])
        self.log(f"Пакет завершён: {len(rows)} файлов, ошибок {errors}")
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить таблицу",
                                              "batch.csv", "CSV (*.csv)")
        if not path:
            return
        batch.save_csv(rows, path)
        self.log(f"Таблица сохранена: {path}")

    def threshold(self):
        return self.threshold_box.value()

    def rebuild_params(self):
        """Пересобирает поля настроек под выбранный детектор."""
        while self.param_row.count():
            item = self.param_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.param_widgets = {}

        detector = self.current_detector()
        for key, label, default in detector.fields:
            spin = QDoubleSpinBox()
            spin.setRange(0.05, 3600.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setValue(float(detector.values.get(key, default)))
            self.param_row.addWidget(QLabel(label))
            self.param_row.addWidget(spin)
            self.param_widgets[key] = spin
        self.param_row.addStretch()

    def log(self, text):
        """Пишет строку в журнал сеанса и в окно."""
        stamp = datetime.now().strftime("%H:%M:%S")
        self.journal.append((stamp, text))
        self.text.append(f"[{stamp}] {text}")

    def current_detector(self):
        return self.detectors[self.detector_box.currentIndex()]

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", FILE_FILTER)
        if not path:
            return
        self.stop_play()
        self.cleanup_temp()
        self.source_path = path
        work_path = path

        if analysis.is_video(path):
            streams = analysis.list_audio_streams(path)
            if not streams:
                self.log("В видеофайле нет звуковых дорожек")
                return
            index = 0
            if len(streams) > 1:
                labels = [
                    f"{s['index']}: {s['codec']}, {s['channels']} кан., "
                    f"{s['sample_rate']} Гц {s['language']}".strip()
                    for s in streams
                ]
                choice, ok = QInputDialog.getItem(
                    self, "Звуковые дорожки", "Какую дорожку исследуем?",
                    labels, 0, False)
                if not ok:
                    return
                index = labels.index(choice)
            self.temp_path = analysis.extract_audio(path, index)
            work_path = self.temp_path

        self.wav_path = work_path
        self.signal, self.sr = analysis.load_audio(work_path)
        self.info = analysis.probe_format(work_path)
        self.times = self.probs = None
        self.service_result = None
        self.results = {}
        self.journal = []
        self.text.clear()

        note = " — сжатие с потерями" if self.info.get("lossy") else ""
        self.status.setText(os.path.basename(path))
        if self.temp_path:
            self.log(
                f"Из видео {os.path.basename(path)} извлечена дорожка {index} "
                f"без перекодирования"
            )
        self.log(
            f"Загружен {os.path.basename(path)}: "
            f"{self.info.get('codec', '?')}{note}, {self.sr} Гц, "
            f"{len(self.signal) / self.sr:.2f} с"
        )
        self.checks = analysis.classic_checks(self.signal, self.sr)
        self.log(
            f"Классические проверки: смещение {self.checks['dc_offset']:.6f}, "
            f"срез {self.checks['cutoff_hz']:.0f} Гц, "
            f"постоянных участков {len(self.checks['constant_runs'])}, "
            f"повторов {len(self.checks['repeats'])}"
        )
        self.draw()

    def run_analysis(self):
        if self.signal is None:
            return
        detector = self.current_detector()
        detector.values = {k: w.value() for k, w in self.param_widgets.items()}
        detector.last_result = None

        settings = ", ".join(
            f"{label} {detector.value(key):g}"
            for key, label, _ in detector.fields
        )
        self.log(f"Запуск детектора: {detector.name}"
                 + (f" ({settings}, порог {self.threshold():g})" if settings else ""))
        self.task = AnalysisTask(detector, self.signal, self.sr, self.wav_path)
        self.task.done.connect(self.on_analysis_done)
        self.task.failed.connect(lambda msg: self.log(f"Ошибка: {msg}"))
        self.task.start()

    def on_analysis_done(self, times, probs):
        detector = self.current_detector()
        self.service_result = detector.last_result

        if len(times) == 0:
            if self.service_result:
                self.log(f"{detector.name}: {self.service_result.get('status')} — "
                         f"{self.service_result.get('reason', '')}")
            else:
                self.log(f"{detector.name}: оценок не получено")
            return

        intervals = analysis.intervals_above(times, probs, self.threshold())
        self.times, self.probs = times, probs
        self.results[detector.name] = {
            "times": times,
            "probs": probs,
            "intervals": intervals,
            "service": self.service_result,
        }

        self.log(
            f"{detector.name}: оценок {len(probs)}, максимум {probs.max():.3f}, "
            f"участков выше порога {len(intervals)}"
        )
        if self.service_result:
            self.log(f"{detector.name}: вердикт сервиса "
                     f"{self.service_result.get('verdict')}")
        for t0, t1 in intervals:
            self.log(f"{detector.name}: участок {t0:.2f} — {t1:.2f} с")
        self.draw()

    def draw(self):
        if self.signal is None:
            return
        self.figure.clear()
        top = self.figure.add_subplot(211)
        power, freqs, times = analysis.spectrogram_db(self.signal, self.sr)
        top.pcolormesh(times, freqs, power, cmap="magma", shading="auto")
        top.set_ylabel("Частота, Гц")

        bottom = self.figure.add_subplot(212, sharex=top)
        if self.probs is not None:
            bottom.plot(self.times, self.probs)
            bottom.axhline(self.threshold(), linestyle="--", color="red")
            for t0, t1 in analysis.intervals_above(self.times, self.probs,
                                                   self.threshold()):
                bottom.axvspan(t0, t1, alpha=0.3, color="red")
        bottom.set_ylim(0, 1)
        bottom.set_xlabel("Время, с")
        bottom.set_ylabel("Оценка")

        self.cursors = [
            top.axvline(0, color="cyan", linewidth=1, visible=False),
            bottom.axvline(0, color="cyan", linewidth=1, visible=False),
        ]

        self.figure.tight_layout()
        self.canvas.draw()

    def save_report(self):
        if self.checks is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчёт",
                                              "report.docx", "DOCX (*.docx)")
        if not path:
            return
        case = {
            "path": self.wav_path,
            "source": self.source_path,
            "signal": self.signal,
            "sr": self.sr,
            "duration": len(self.signal) / self.sr,
            "info": self.info,
            "checks": self.checks,
            "threshold": self.threshold(),
            "results": self.results,
            "journal": self.journal,
        }
        report.build_report(path, case)
        self.log(f"Отчёт сохранён: {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
