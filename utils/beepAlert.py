import numpy as np
import simpleaudio as sa
import threading

def _beep(freq=1000, duration=0.2):
    fs = 44100
    t = np.linspace(0, duration, int(fs * duration), False)
    wave = np.sin(freq * t * 2 * np.pi)
    audio = (wave * 32767).astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()

def beep_alerta(freq=1000, duration=0.2):
    threading.Thread(target=_beep, args=(freq, duration), daemon=True).start()
