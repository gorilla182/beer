"""
🍺 Beer Allure Reporter
=======================
Открывай пиво — тесты запустятся сами, а потом откроется отчёт!

Цепочка при звуке открытия банки:
  1. pytest --alluredir=allure-results (запуск тестов)
  2. allure serve allure-results        (отчёт в браузере)

Зависимости:
    pip install sounddevice numpy

Запуск:
    python beer_allure.py
"""

import os
import subprocess
import sys
import time
import threading
import numpy as np

# =============================================================================
# 🔧 НАСТРОЙ ЭТИ ДВЕ ПЕРЕМЕННЫЕ ПОД СЕБЯ
# =============================================================================

# Путь к корню проекта с автотестами (можно использовать ~)
# Пример: "~/PycharmProjects/GoldApple"
AUTOTESTS_PROJECT_PATH = "~/PycharmProjects/GoldApple"

# Папка для результатов allure (относительно проекта или абсолютная)
# Пример: "allure-results"  или  "~/PycharmProjects/GoldApple/allure-results"
ALLURE_RESULTS_DIR = "allure-results"

# =============================================================================
# ⚙️  ПАРАМЕТРЫ ДЕТЕКТОРА ЗВУКА (можно не трогать)
# =============================================================================

SAMPLE_RATE          = 44100
CHUNK_SIZE           = int(SAMPLE_RATE * 0.05)
AMPLITUDE_THRESHOLD  = 0.040   # порог громкости (0.0–1.0)
LOUD_SAMPLE_RATIO    = 0.3    # доля громких семплов в чанке
COOLDOWN_SECONDS     = 30     # пауза между срабатываниями (тесты идут долго)
HISS_FREQ_LOW        = 2000   # нижняя граница «шипения» банки (Гц)
HISS_FREQ_HIGH       = 8000   # верхняя граница «шипения» банки (Гц)
HISS_RATIO_THRESHOLD = 0.335   # доля шипящих частот в спектре

# =============================================================================

_last_trigger_time = 0.0
_lock = threading.Lock()
_pipeline_running = False


def expand(path: str) -> str:
    """Раскрывает ~ и переменные окружения в пути."""
    return os.path.expandvars(os.path.expanduser(path))


def check_paths() -> bool:
    ok = True
    project = expand(AUTOTESTS_PROJECT_PATH)

    if not os.path.isdir(project):
        print(f"⚠️  AUTOTESTS_PROJECT_PATH не найден: {project}")
        ok = False

    # ALLURE_RESULTS_DIR может быть относительным — проверяем только если абсолютный
    results = expand(ALLURE_RESULTS_DIR)
    if os.path.isabs(results) and not os.path.isdir(results):
        print(f"⚠️  ALLURE_RESULTS_DIR не найден: {results}")
        print("    (папка создастся автоматически после первого запуска pytest)")

    return ok


def run_pipeline() -> None:
    """Запускает pytest, затем allure serve."""
    global _pipeline_running
    _pipeline_running = True

    project = expand(AUTOTESTS_PROJECT_PATH)
    results = expand(ALLURE_RESULTS_DIR)

    # Если путь относительный — делаем его абсолютным относительно проекта
    if not os.path.isabs(results):
        results = os.path.join(project, results)

    print("\n" + "=" * 55)
    print("  🍺  ПИВО ОТКРЫТО! Поехали...")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Шаг 1: pytest
    # ------------------------------------------------------------------
    print(f"\n🧪  Шаг 1/2 — Запускаю тесты...\n")
    pytest_cmd = [
        sys.executable, "-m", "pytest",
        f"--alluredir={results}",
        "--clean-alluredir",   # очищаем старые результаты перед прогоном
        "-v",
    ]
    print(f"    $ {' '.join(pytest_cmd)}\n")

    pytest_result = subprocess.run(
        pytest_cmd,
        cwd=project,
    )

    exit_code = pytest_result.returncode
    if exit_code == 0:
        print("\n✅  Тесты прошли успешно!")
    elif exit_code == 1:
        print("\n⚠️  Есть упавшие тесты — смотри отчёт.")
    else:
        print(f"\n❌  pytest завершился с кодом {exit_code}.")

    # ------------------------------------------------------------------
    # Шаг 2: allure serve
    # ------------------------------------------------------------------
    print(f"\n📊  Шаг 2/2 — Открываю Allure-отчёт...\n")

    allure_cmd = "allure.bat" if sys.platform.startswith("win") else "allure"
    serve_cmd  = [allure_cmd, "serve", results]
    print(f"    $ {' '.join(serve_cmd)}\n")

    try:
        # allure serve сам открывает браузер и держит сервер — запускаем в фоне
        subprocess.Popen(
            serve_cmd,
            cwd=project,
        )
        print("✅  Allure запущен! Отчёт откроется в браузере автоматически.")
        print("    Останови сервер вручную (Ctrl+C в его окне) когда закончишь.\n")
    except FileNotFoundError:
        print("❌  Команда 'allure' не найдена.")
        print("    Установи Allure CLI: https://allurereport.org/docs/install/")

    _pipeline_running = False


def is_beer_sound(audio_chunk: np.ndarray) -> bool:
    abs_chunk  = np.abs(audio_chunk)
    loud_ratio = np.mean(abs_chunk > AMPLITUDE_THRESHOLD)
    if loud_ratio < LOUD_SAMPLE_RATIO:
        return False

    fft_vals     = np.abs(np.fft.rfft(audio_chunk))
    freqs        = np.fft.rfftfreq(len(audio_chunk), d=1.0 / SAMPLE_RATE)
    total_energy = np.sum(fft_vals) + 1e-9
    hiss_mask    = (freqs >= HISS_FREQ_LOW) & (freqs <= HISS_FREQ_HIGH)
    hiss_ratio   = np.sum(fft_vals[hiss_mask]) / total_energy

    return hiss_ratio >= HISS_RATIO_THRESHOLD


def audio_callback(indata, frames, time_info, status) -> None:
    global _last_trigger_time

    if _pipeline_running:
        return  # тесты уже идут — не реагируем

    mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
    if not is_beer_sound(mono):
        return

    with _lock:
        now = time.monotonic()
        if now - _last_trigger_time < COOLDOWN_SECONDS:
            return
        _last_trigger_time = now

    threading.Thread(target=run_pipeline, daemon=True).start()


def main() -> None:
    print("=" * 55)
    print("  🍺  Beer Allure Reporter  🍺")
    print("=" * 55)

    if not check_paths():
        print("\n❌  Исправь пути в начале скрипта и запусти снова.")
        sys.exit(1)

    print(f"\n📁  Проект:   {expand(AUTOTESTS_PROJECT_PATH)}")
    print(f"📂  Results:  {expand(ALLURE_RESULTS_DIR)}")
    print(f"\n🎙️  Слушаю микрофон... Открывай пиво!\n")
    print("    (Ctrl+C для выхода)\n")

    try:
        import sounddevice as sd
    except ImportError:
        print("❌  Установи sounddevice:  pip install sounddevice numpy")
        sys.exit(1)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n👋  Выход. Пей ответственно!")
    except Exception as exc:
        print(f"\n❌  Ошибка аудиопотока: {exc}")
        print("    Проверь, что микрофон подключён и не занят другим приложением.")
        sys.exit(1)


if __name__ == "__main__":
    main()