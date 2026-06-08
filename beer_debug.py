"""
🍺 Beer Debug Tool
==================
Запусти этот скрипт, затем открой банку пива рядом с микрофоном.
Скрипт покажет реальные значения амплитуды и hiss_ratio —
скопируй их в beer_allure.py как пороги.

Запуск:
    python beer_debug.py
"""

import sys
import time
import numpy as np

SAMPLE_RATE   = 44100
CHUNK_DURATION = 0.05
CHUNK_SIZE    = int(SAMPLE_RATE * CHUNK_DURATION)
HISS_FREQ_LOW  = 2000
HISS_FREQ_HIGH = 8000

max_amp_seen   = 0.0
max_hiss_seen  = 0.0

def audio_callback(indata, frames, time_info, status):
    global max_amp_seen, max_hiss_seen

    mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
    amp  = float(np.max(np.abs(mono)))

    fft_vals     = np.abs(np.fft.rfft(mono))
    freqs        = np.fft.rfftfreq(len(mono), d=1.0 / SAMPLE_RATE)
    total_energy = np.sum(fft_vals) + 1e-9
    hiss_mask    = (freqs >= HISS_FREQ_LOW) & (freqs <= HISS_FREQ_HIGH)
    hiss_ratio   = float(np.sum(fft_vals[hiss_mask]) / total_energy)

    # Обновляем рекорды
    if amp > max_amp_seen:
        max_amp_seen = amp
    if hiss_ratio > max_hiss_seen:
        max_hiss_seen = hiss_ratio

    # Выводим только если хоть что-то слышно
    if amp > 0.01:
        bar_amp  = "█" * int(amp * 40)
        bar_hiss = "█" * int(hiss_ratio * 40)
        print(f"\r  амплитуда: {amp:.3f}  {bar_amp:<40}  |  "
              f"hiss: {hiss_ratio:.3f}  {bar_hiss:<40}", end="", flush=True)

def main():
    print("=" * 70)
    print("  🍺  Beer Debug Tool  — открой банку рядом с микрофоном")
    print("=" * 70)
    print()
    print("  Сейчас тихо? Запомни фоновые значения амплитуды и hiss.")
    print("  Затем открой пиво — посмотри на пики.")
    print("  Нажми Ctrl+C когда наоргался — увидишь итог.\n")

    try:
        import sounddevice as sd
    except ImportError:
        print("❌  pip install sounddevice numpy")
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
        print(f"\n\n{'='*70}")
        print(f"  📊  МАКСИМАЛЬНЫЕ ЗНАЧЕНИЯ ЗА СЕССИЮ:")
        print(f"      амплитуда (пик):  {max_amp_seen:.3f}")
        print(f"      hiss_ratio (пик): {max_hiss_seen:.3f}")
        print(f"{'='*70}")
        print()
        print(f"  Скопируй эти значения в beer_allure.py:")
        print(f"  (ставь чуть ниже пика, чтобы срабатывало)")
        print()

        amp_suggestion  = round(max_amp_seen  * 0.6, 2)
        hiss_suggestion = round(max_hiss_seen * 0.6, 2)
        print(f"  AMPLITUDE_THRESHOLD   = {amp_suggestion}")
        print(f"  LOUD_SAMPLE_RATIO     = 0.3   # обычно не меняй")
        print(f"  HISS_RATIO_THRESHOLD  = {hiss_suggestion}")
        print()

def check_microphone():
    """Проверяет доступность микрофона."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        inputs = [d for d in devices if d['max_input_channels'] > 0]
        if not inputs:
            print("❌  Микрофон не найден!")
            sys.exit(1)
        print("  🎙️  Доступные микрофоны:")
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                marker = " ◀ (по умолчанию)" if i == sd.default.device[0] else ""
                print(f"      [{i}] {d['name']}{marker}")
        print()
    except Exception as e:
        print(f"❌  Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_microphone()
    main()