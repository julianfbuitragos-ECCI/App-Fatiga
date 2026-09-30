"""
vibration_analysis.py
----------------------
Análisis espectral de la señal de vibración: FFT + extracción de
características + clasificación del tipo de falla (normal, pista interna,
pista externa, elemento rodante), inspirado en el diagnóstico clásico de
fallas de rodamientos (frecuencias BPFI, BPFO, BSF).

IMPORTANTE (léelo para tu documento de tesis):
Igual que en image_analysis.py, el modelo se entrena con señales
SINTÉTICAS generadas a partir de las frecuencias características típicas
de cada tipo de falla (como proporción de la frecuencia de giro). Para una
versión con validez experimental real, reemplaza
`_generate_synthetic_signals()` cargando señales reales (por ejemplo del
dataset público CWRU) y vuelve a entrenar con `train_vibration_model()`.
"""

from pathlib import Path

import numpy as np
import joblib
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = Path(__file__).parent / "models" / "vibration_model.joblib"
VIBRATION_CLASSES = ["normal", "pista_interna", "pista_externa", "elemento_rodante"]
FEATURE_NAMES = ["dominant_freq_ratio", "spectral_centroid", "rms", "kurtosis", "crest_factor"]

# Multiplicadores típicos de frecuencia característica de falla respecto a
# la frecuencia de giro del eje (valores de referencia usados en literatura
# de diagnóstico de rodamientos; varían según la geometría real del rodamiento).
FAULT_FREQ_RATIO = {
    "pista_interna": 5.4,
    "pista_externa": 3.6,
    "elemento_rodante": 2.4,
}


# ---------------------------------------------------------------------------
# Extracción de características
# ---------------------------------------------------------------------------

def extract_spectral_features(signal: np.ndarray, fs: float, shaft_freq: float) -> np.ndarray:
    """Extrae características del espectro de una señal de vibración.

    signal: arreglo 1D con la señal en el tiempo.
    fs: frecuencia de muestreo (Hz).
    shaft_freq: frecuencia de giro del eje (Hz), usada para normalizar el
                pico dominante y hacerlo comparable entre condiciones.
    """
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - signal.mean()

    n = len(signal)
    freqs = rfftfreq(n, d=1 / fs)
    spectrum = np.abs(rfft(signal))

    # Ignorar la componente DC y frecuencias por debajo de 0.5x la de giro
    mask = freqs > (0.5 * shaft_freq)
    freqs_m, spectrum_m = freqs[mask], spectrum[mask]

    dominant_freq = freqs_m[np.argmax(spectrum_m)] if len(freqs_m) else 0.0
    dominant_freq_ratio = dominant_freq / shaft_freq if shaft_freq else 0.0

    spectral_centroid = (
        np.sum(freqs_m * spectrum_m) / np.sum(spectrum_m) if np.sum(spectrum_m) > 0 else 0.0
    ) / shaft_freq if shaft_freq else 0.0

    rms = np.sqrt(np.mean(signal ** 2))
    kurt = kurtosis(signal, fisher=True)  # impulsividad: alta en fallas localizadas
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms if rms > 0 else 0.0

    return np.array([dominant_freq_ratio, spectral_centroid, rms, kurt, crest_factor],
                     dtype=np.float32)


# ---------------------------------------------------------------------------
# Datos sintéticos de entrenamiento (ver aviso arriba)
# ---------------------------------------------------------------------------

def _simulate_signal(fault: str, fs: float, duration: float, shaft_freq: float,
                      rng: np.random.Generator) -> np.ndarray:
    t = np.arange(0, duration, 1 / fs)
    # Componente base: giro del eje + un armónico
    signal = 0.5 * np.sin(2 * np.pi * shaft_freq * t)
    signal += 0.15 * np.sin(2 * np.pi * 2 * shaft_freq * t)

    if fault != "normal":
        fault_freq = FAULT_FREQ_RATIO[fault] * shaft_freq
        # Tren de impactos periódicos moduladas (típico de defectos localizados)
        impact_train = np.sin(2 * np.pi * fault_freq * t)
        envelope = (impact_train > 0.95).astype(float)  # picos angostos = impulsivo
        amplitude = {"pista_interna": 1.2, "pista_externa": 0.9, "elemento_rodante": 1.0}[fault]
        signal += amplitude * envelope * np.sin(2 * np.pi * fault_freq * 3 * t)

    signal += rng.normal(0, 0.08, size=t.shape)  # ruido de fondo
    return signal


def _generate_synthetic_dataset(n_per_class: int = 120, seed: int = 7):
    rng = np.random.default_rng(seed)
    fs = 12_000.0  # Hz, igual orden de magnitud que datasets reales tipo CWRU
    duration = 0.5  # segundos por muestra

    X, y = [], []
    for fault in VIBRATION_CLASSES:
        for _ in range(n_per_class):
            shaft_freq = rng.uniform(28, 32)  # ~1740-1920 rpm, rango típico de motores de prueba
            sig = _simulate_signal(fault, fs, duration, shaft_freq, rng)
            feats = extract_spectral_features(sig, fs, shaft_freq)
            X.append(feats)
            y.append(fault)
    return np.vstack(X), np.array(y)


def train_vibration_model():
    X, y = _generate_synthetic_dataset()
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=7)
    clf.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    return clf


def load_vibration_model():
    if not MODEL_PATH.exists():
        return train_vibration_model()
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# API principal del módulo
# ---------------------------------------------------------------------------

def analyze_vibration(signal: np.ndarray, fs: float, shaft_freq: float) -> dict:
    features = extract_spectral_features(signal, fs, shaft_freq)
    clf = load_vibration_model()

    proba = clf.predict_proba(features.reshape(1, -1))[0]
    pred_idx = int(np.argmax(proba))
    pred_class = clf.classes_[pred_idx]
    confidence = float(proba[pred_idx])

    freqs = rfftfreq(len(signal), d=1 / fs)
    spectrum = np.abs(rfft(np.asarray(signal) - np.mean(signal)))

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": dict(zip(clf.classes_, proba.tolist())),
        "features": dict(zip(FEATURE_NAMES, features.tolist())),
        "freqs": freqs,
        "spectrum": spectrum,
    }


def generate_demo_signal(fault: str, fs: float = 12_000.0, duration: float = 0.5,
                          shaft_freq: float = 30.0, seed: int | None = None) -> np.ndarray:
    """Genera una señal de demostración para probar la app sin sensor real."""
    rng = np.random.default_rng(seed)
    return _simulate_signal(fault, fs, duration, shaft_freq, rng)


if __name__ == "__main__":
    demo = generate_demo_signal("pista_interna", seed=1)
    result = analyze_vibration(demo, fs=12_000.0, shaft_freq=30.0)
    print("Clase predicha:", result["predicted_class"])
    print("Confianza:", round(result["confidence"], 3))
    print("Probabilidades:", {k: round(v, 3) for k, v in result["probabilities"].items()})
