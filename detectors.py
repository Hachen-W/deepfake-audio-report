"""Детекторы синтезированной речи с общим интерфейсом.

Каждый детектор получает сигнал и возвращает кривую оценок во времени,
поэтому интерфейс приложения одинаково работает с любым из них.
Модели, которые дают одну оценку на весь файл, возвращают ровную линию.

Поле `fields` описывает настройки детектора: интерфейс строит по нему поля
ввода, а перед запуском кладёт введённые значения в `values`.
"""
import os

import numpy as np

import analysis
import config


class Detector:
    name = "детектор"
    fields = []     # (ключ, подпись, значение по умолчанию)

    def __init__(self):
        self.values = {}
        self.last_result = None

    def value(self, key):
        """Значение настройки: введённое пользователем или из умолчаний."""
        for field_key, _, default in self.fields:
            if field_key == key:
                return float(self.values.get(key, default))
        raise KeyError(key)

    def scores(self, x, sr, path):
        """Возвращает (times, probs) — оценки по времени."""
        raise NotImplementedError


class BaselineDetector(Detector):
    """ЗАГЛУШКА, а не детектор. Две спектральные характеристики, чтобы
    интерфейс работал без моделей. Результаты нельзя выдавать
    за детектирование."""
    name = "ЗАГЛУШКА (локально)"
    fields = [
        ("window", "Окно, с", config.WINDOW),
        ("hop", "Шаг, с", config.HOP),
    ]

    def scores(self, x, sr, path):
        return analysis.frame_probabilities(
            x, sr, None, self.value("window"), self.value("hop"))


class OnnxDetector(Detector):
    """Локальная модель в формате ONNX."""
    fields = BaselineDetector.fields

    def __init__(self, path=config.MODEL_PATH):
        super().__init__()
        import onnxruntime
        self.session = onnxruntime.InferenceSession(path)
        self.name = f"ONNX локально ({os.path.basename(path)})"

    def scores(self, x, sr, path):
        return analysis.frame_probabilities(
            x, sr, self.session, self.value("window"), self.value("hop"))


class SrwStreamDetector(Detector):
    """Модель сервиса в потоковом режиме: даёт кривую по всему файлу."""
    fields = [
        ("chunk", "Чанк, с", config.CHUNK_SECONDS),
        ("idle", "Тишина, с", config.STREAM_IDLE),
        ("timeout", "Таймаут, с", config.STREAM_TIMEOUT),
    ]

    def __init__(self, model, title):
        super().__init__()
        self.model = model
        self.name = title

    def scores(self, x, sr, path):
        import service
        return service.ServiceClient().stream_scores(
            x, sr, self.model,
            chunk_seconds=self.value("chunk"),
            idle=self.value("idle"),
            timeout=self.value("timeout"),
        )


class SrwQueueDetector(Detector):
    """Модель сервиса через очередь: одна оценка на весь файл.

    Рисуется ровной линией, потому что привязки ко времени сервис не даёт.
    """
    fields = [
        ("poll", "Опрос, с", config.POLL_INTERVAL),
        ("timeout", "Таймаут, с", config.RESULT_TIMEOUT),
        ("retry", "Пауза 429, с", config.RETRY_PAUSE),
    ]

    def __init__(self, model, title):
        super().__init__()
        self.model = model
        self.name = title

    def scores(self, x, sr, path):
        import service
        result = service.ServiceClient().analyze(
            path, self.model,
            poll=self.value("poll"),
            timeout=self.value("timeout"),
            retry=self.value("retry"),
        )
        self.last_result = result
        if result.get("status") != "completed":
            return np.array([]), np.array([])
        value = float(result["prediction"])
        duration = len(x) / sr
        return np.array([0.0, duration]), np.array([value, value])


def available():
    """Детекторы, которые можно выбрать в интерфейсе."""
    result = [BaselineDetector()]
    if os.path.exists(config.MODEL_PATH):
        result.append(OnnxDetector())
    result += [
        SrwStreamDetector("pytorch", "SRW: PyTorch — поток"),
        SrwStreamDetector("pyara", "SRW: PyAra — поток"),
        SrwQueueDetector("pytorch", "SRW: PyTorch — файл целиком"),
        SrwQueueDetector("pyara", "SRW: PyAra — файл целиком"),
    ]
    return result
