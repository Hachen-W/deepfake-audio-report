"""Воспроизведение фонограммы внешним проигрывателем.

QtMultimedia не используется: её FFmpeg-бэкенд падает на некоторых сборках,
а сегфолт из питона не перехватить. Играем сигнал через ffplay отдельным
процессом — так приложение не может упасть вместе с проигрывателем.
"""
import os
import shutil
import subprocess
import tempfile
import time
import wave

import numpy as np


def write_wav(path, x, sr):
    """Сохраняет сигнал во временный wav для проигрывателя."""
    data = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sr))
        wav_file.writeframes(data)


class Player:
    def __init__(self):
        self.process = None
        self.temp_path = None
        self.started = 0.0
        self.duration = 0.0

    def play(self, x, sr):
        """Запускает воспроизведение с начала записи."""
        self.stop()

        player_bin = shutil.which("ffplay") or shutil.which("aplay")
        if player_bin is None:
            raise RuntimeError("Не найден ffplay или aplay для воспроизведения")

        handle, self.temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
        write_wav(self.temp_path, x, sr)

        if player_bin.endswith("ffplay"):
            command = [player_bin, "-hide_banner", "-loglevel", "quiet",
                       "-nodisp", "-autoexit", self.temp_path]
        else:
            command = [player_bin, "-q", self.temp_path]

        self.process = subprocess.Popen(command)
        self.started = time.monotonic()
        self.duration = len(x) / sr

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

        if self.temp_path and os.path.exists(self.temp_path):
            os.remove(self.temp_path)
        self.temp_path = None

    def is_playing(self):
        return self.process is not None and self.process.poll() is None

    def position(self):
        """Позиция в секундах — по прошедшему времени с момента запуска."""
        if self.process is None:
            return 0.0
        return min(time.monotonic() - self.started, self.duration)
