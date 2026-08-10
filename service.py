"""Клиент к сервису SRW: логин, отправка файла, ожидание результата."""
import json
import math
import os
import subprocess
import tempfile
import time

import numpy as np
import requests
from scipy.signal import resample_poly
from websocket import WebSocketTimeoutException, create_connection

import analysis
import config

# Сервис принимает только эти расширения (ALLOWED_EXTENSIONS в routes/audio.py)
SERVICE_EXTENSIONS = {".wav", ".mp3", ".aac", ".flac", ".ogg"}


def prepare_for_service(path):
    """Готовит файл к отправке. Возвращает (путь, временный ли файл).

    Форматы из списка уходят как есть — без пересжатия и без изменения
    частоты, чтобы модель получила ровно исходный сигнал. Остальное
    (opus, wma, дорожка из видео в контейнере mka) декодируется в wav
    с сохранением частоты и каналов.
    """
    if os.path.splitext(path)[1].lower() in SERVICE_EXTENSIONS:
        return path, False

    exe = analysis.ffmpeg_bin("ffmpeg")
    if exe is None:
        raise RuntimeError("Сервис не принимает этот формат, а ffmpeg не найден")

    handle, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    subprocess.run(
        [exe, "-v", "quiet", "-y", "-i", path,
         "-acodec", "pcm_s16le", "-vn", out_path], check=True)
    return out_path, True


def to_pcm16(x, sr):
    """Приводит сигнал к 16 кГц моно int16 — в таком виде его ждёт воркер."""
    if sr != config.STREAM_SR:
        g = math.gcd(int(sr), config.STREAM_SR)
        x = resample_poly(x, config.STREAM_SR // g, int(sr) // g)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16).tobytes()


class ServiceClient:
    def __init__(self, base_url=config.BASE_URL,
                 username=config.USERNAME, password=config.PASSWORD):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.token = None

    def login(self):
        """Получает access-токен. Он живёт 2 минуты, поэтому логин повторяется."""
        response = requests.post(
            f"{self.base_url}/auth/login",
            json={"username": self.username, "password": self.password},
            timeout=10,
        )
        if response.status_code == 401:
            raise RuntimeError("Неверный логин или пароль")
        response.raise_for_status()
        self.token = response.json()["access_token"]

    def headers(self):
        if self.token is None:
            self.login()
        return {"Authorization": f"Bearer {self.token}"}

    def post_file(self, file_path, model):
        """Одна попытка отправки файла."""
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            return requests.post(
                f"{self.base_url}/audio/detect",
                headers=self.headers(), files=files,
                data={"model": model}, timeout=60,
            )

    def detect(self, file_path, model=config.DEFAULT_MODEL,
               retry=config.RETRY_PAUSE, attempts=3):
        """Отправляет фонограмму, возвращает request_id."""
        for _ in range(attempts):
            response = self.post_file(file_path, model)
            if response.status_code == 401:
                self.login()                      # токен протух
                continue
            if response.status_code == 429:
                time.sleep(retry)                 # лимит запросов исчерпан
                continue
            if response.status_code == 403:
                raise RuntimeError("Нужна роль SERVICE или ADMIN")
            if response.status_code == 413:
                raise RuntimeError(
                    "Файл больше лимита сервиса. Поднимите MAX_FILE_SIZE_MB "
                    "в docker-compose и перезапустите api")
            response.raise_for_status()
            return response.json()["request_id"]
        raise RuntimeError("Сервис не принял файл")

    def wait_result(self, request_id, poll=config.POLL_INTERVAL,
                    timeout=config.RESULT_TIMEOUT, retry=config.RETRY_PAUSE):
        """Опрашивает сервис, пока не появится результат."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            response = requests.get(
                f"{self.base_url}/audio/result/{request_id}",
                headers=self.headers(), timeout=10,
            )
            if response.status_code == 401:
                self.login()
                continue
            if response.status_code == 429:
                # Роли USER и ADMIN дают 10 запросов в минуту, ждём окно
                time.sleep(config.RETRY_PAUSE)
                continue
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "processing":
                return data
            time.sleep(poll)
        raise TimeoutError("Сервис не ответил за отведённое время")

    def stream_scores(self, x, sr, model=config.DEFAULT_MODEL,
                      chunk_seconds=config.CHUNK_SECONDS,
                      idle=config.STREAM_IDLE, timeout=config.STREAM_TIMEOUT):
        """Гонит фонограмму через websocket, возвращает (times, scores).

        Оценки приходят по мере обработки, поэтому время каждой точки берём
        по количеству уже отправленного звука.
        """
        if self.token is None:
            self.login()

        url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws = create_connection(
            f"{url}/audio/stream?token={self.token}&model={model}", timeout=15)

        pcm = to_pcm16(x, sr)
        step = int(config.STREAM_SR * chunk_seconds) * 2   # 2 байта на отсчёт
        times, scores = [], []

        def drain(position, timeout):
            """Забирает всё, что воркер успел прислать."""
            ws.settimeout(timeout)
            while True:
                try:
                    data = json.loads(ws.recv())
                except (WebSocketTimeoutException, ValueError):
                    return True
                except Exception:
                    return False
                # Позицию сообщает воркер: он может сильно отставать от отправки
                times.append(data.get("position", position))
                scores.append(data.get("current_score", 0.0))
                if data.get("status") == "terminated":
                    return False

        duration = len(pcm) / 2 / config.STREAM_SR
        try:
            alive = True
            for start in range(0, len(pcm), step):
                ws.send_binary(pcm[start:start + step])
                position = (start + step) / 2 / config.STREAM_SR
                if not drain(position, 0.05):
                    alive = False
                    break

            # Медленные модели отвечают заметно позже, чем мы отправили звук,
            # поэтому ждём, пока ответы не перестанут приходить.
            deadline = time.time() + timeout
            idle_until = time.time() + idle
            while alive and time.time() < deadline and time.time() < idle_until:
                before = len(scores)
                if not drain(duration, 1.0):
                    break
                if len(scores) > before:
                    idle_until = time.time() + idle
        finally:
            try:
                ws.close()
            except Exception:
                pass

        return np.array(times), np.array(scores)

    def analyze(self, file_path, model=config.DEFAULT_MODEL,
                poll=config.POLL_INTERVAL, timeout=config.RESULT_TIMEOUT,
                retry=config.RETRY_PAUSE):
        """Полный цикл через очередь: отправить и дождаться вердикта."""
        send_path, temporary = prepare_for_service(file_path)
        try:
            request_id = self.detect(send_path, model, retry)
            return self.wait_result(request_id, poll, timeout, retry)
        finally:
            if temporary and os.path.exists(send_path):
                os.remove(send_path)
