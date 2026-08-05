"""Аналитическое ядро демо-прототипа."""
import json
import shutil
import subprocess

import numpy as np

# Кодеки со сжатием с потерями. Для них часть проверок неинформативна.
LOSSY_CODECS = {"mp3", "aac", "wmav1", "wmav2", "vorbis", "opus",
                "amr_nb", "amr_wb", "gsm", "ac3", "atrac3"}


def ffmpeg_bin(name="ffmpeg"):
    """Ищет ffmpeg/ffprobe в системе, иначе берёт из пакета imageio-ffmpeg."""
    path = shutil.which(name)
    if path:
        return path
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    return None


def probe_format(path):
    """Сведения о контейнере и кодеке через ffprobe. Идут в отчёт."""
    exe = ffmpeg_bin("ffprobe")
    if exe is None:
        return {}
    cmd = [exe, "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", path]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    data = json.loads(out)
    audio = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        return {}
    stream = audio[0]
    codec = stream.get("codec_name", "")
    return {
        "container": data.get("format", {}).get("format_name", ""),
        "codec": codec,
        "sample_rate": int(stream.get("sample_rate", 0)),
        "channels": stream.get("channels", 0),
        "bit_rate": data.get("format", {}).get("bit_rate", ""),
        "lossy": codec in LOSSY_CODECS,
    }


def load_audio(path):
    """Читает аудио любого формата, возвращает (моно float, частота)."""
    try:
        import soundfile as sf
        x, sr = sf.read(path, always_2d=True, dtype="float64")
        return x.mean(axis=1), sr
    except Exception:
        return load_via_ffmpeg(path)


def load_via_ffmpeg(path):
    """Резерв для того, что не читает libsndfile: m4a, wma, amr, opus и прочее."""
    exe = ffmpeg_bin("ffmpeg")
    if exe is None:
        raise RuntimeError(
            "Файл не прочитан: нет ни soundfile, ни ffmpeg. "
            "Выполните ./run.sh — он поставит зависимости сам.")
    info = probe_format(path)
    sr = info.get("sample_rate") or 44100
    cmd = [exe, "-v", "quiet", "-i", path,
           "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64), sr


# ---------- детектор синтеза ----------

def baseline_score(seg, sr):
    """ЗАГЛУШКА, а не детектор.

    Считает две простые спектральные характеристики. Нужна только чтобы
    интерфейс работал до подключения настоящей модели. Числа отсюда
    нельзя показывать как результат детектирования.
    """
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    total = spec.sum() + 1e-9
    high = spec[freqs > 6000].sum() / total          # доля ВЧ-энергии
    flat = spec.std() / (spec.mean() + 1e-9)         # изрезанность спектра
    score = 0.5 + 0.5 * (0.3 - high) - 0.05 * flat
    return float(np.clip(score, 0.0, 1.0))


def onnx_score(session, seg, sr):
    """Один сегмент через ONNX-модель. Вход подгоняется под конкретную модель."""
    inp = session.get_inputs()[0]
    data = seg.astype(np.float32)[None, :]
    out = session.run(None, {inp.name: data})[0]
    out = np.asarray(out).ravel()
    if out.size == 1:
        return float(1 / (1 + np.exp(-out[0])))
    e = np.exp(out - out.max())
    return float((e / e.sum())[-1])


def frame_probabilities(x, sr, session=None, win=2.0, hop=0.5):
    """Скользящая оценка вероятности синтеза. Возвращает (times, probs)."""
    n = int(win * sr)
    step = int(hop * sr)
    times, probs = [], []
    for start in range(0, max(len(x) - n + 1, 1), step):
        seg = x[start:start + n]
        if len(seg) < n:
            break
        p = onnx_score(session, seg, sr) if session else baseline_score(seg, sr)
        times.append((start + n / 2) / sr)
        probs.append(p)
    return np.array(times), np.array(probs)


def intervals_above(times, probs, threshold=0.7):
    """Склеивает соседние точки выше порога в интервалы [(t0, t1), ...]."""
    result = []
    start = None
    for t, p in zip(times, probs):
        if p >= threshold and start is None:
            start = t
        elif p < threshold and start is not None:
            result.append((start, t))
            start = None
    if start is not None:
        result.append((start, times[-1]))
    return result


# ---------- классические проверки аутентичности ----------

def dc_offset(x):
    """Смещение постоянной составляющей."""
    return float(x.mean())


def spectral_cutoff(x, sr):
    """Частота среза спектра. Резкий срез ниже Найквиста — признак lossy-кодека."""
    spec = np.abs(np.fft.rfft(x[:sr * 10] if len(x) > sr * 10 else x))
    freqs = np.fft.rfftfreq(len(x[:sr * 10] if len(x) > sr * 10 else x), 1 / sr)
    energy = np.cumsum(spec ** 2)
    energy = energy / energy[-1]
    idx = int(np.searchsorted(energy, 0.995))
    return float(freqs[min(idx, len(freqs) - 1)])


def constant_runs(x, sr, min_ms=20):
    """Ищет участки с постоянным значением отсчётов (дропауты, вставки)."""
    same = np.abs(np.diff(x)) < 1e-6
    result = []
    start = None
    for i, flag in enumerate(same):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if (i - start) / sr * 1000 >= min_ms:
                result.append((start / sr, i / sr))
            start = None
    return result


def find_repeats(x, sr, block=0.5, threshold=0.999):
    """Ищет повторяющиеся фрагменты по спектральным отпечаткам блоков."""
    n = int(block * sr)
    prints = []
    for start in range(0, len(x) - n + 1, n):
        seg = x[start:start + n]
        spec = np.abs(np.fft.rfft(seg))[:256]
        norm = np.linalg.norm(spec) + 1e-9
        prints.append((start / sr, spec / norm))

    pairs = []
    for i in range(len(prints)):
        for j in range(i + 2, len(prints)):
            if float(prints[i][1] @ prints[j][1]) >= threshold:
                pairs.append((prints[i][0], prints[j][0]))
    return pairs


def classic_checks(x, sr):
    """Собирает все классические проверки в один словарь."""
    return {
        "dc_offset": dc_offset(x),
        "cutoff_hz": spectral_cutoff(x, sr),
        "constant_runs": constant_runs(x, sr),
        "repeats": find_repeats(x, sr),
    }
