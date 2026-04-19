import numpy as np
import nolds
from scipy.signal import welch
import warnings # <-- НОВОЕ
from typing import Dict, Union

def extract_features(rr_intervals: np.ndarray, calc_nonlinear: bool = True) -> Dict[str, Union[float, int]]:

    # Вычисляет вектор признаков для ряда RR-интервалов.
    # calc_nonlinear (bool): Если False, пропускает тяжелые вычисления (для Fast Mode).

    features = {}

    # Временные метрики (Time Domain)
    features['mean_rr'] = np.mean(rr_intervals)
    features['sdnn'] = np.std(rr_intervals, ddof=1)
    features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))

    # Частотные метрики (Frequency Domain)
    try:
        f, pxx = welch(rr_intervals, fs=1.0, nperseg=min(len(rr_intervals), 256))
        lf_band, hf_band = (0.04, 0.15), (0.15, 0.40)
        lf_power = np.trapezoid(pxx[(f >= lf_band[0]) & (f <= lf_band[1])], f[(f >= lf_band[0]) & (f <= lf_band[1])])
        hf_power = np.trapezoid(pxx[(f >= hf_band[0]) & (f <= hf_band[1])], f[(f >= hf_band[0]) & (f <= hf_band[1])])

        features['lf_power'] = lf_power
        features['hf_power'] = hf_power
        features['lf_hf_ratio'] = lf_power / hf_power if hf_power > 0 else 0.0
    except Exception:
        features['lf_power'] = 0.0
        features['hf_power'] = 0.0
        features['lf_hf_ratio'] = 0.0

    # Нелинейные метрики (Nonlinear / Takens)
    if calc_nonlinear:
        # Игнорируем специфичные RuntimeWarning от nolds (плоские участки)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                features['sampen'] = nolds.sampen(rr_intervals, emb_dim=2)
                features['corr_dim'] = nolds.corr_dim(rr_intervals, emb_dim=2)
            except Exception:
                features['sampen'] = 0.0
                features['corr_dim'] = 0.0
    else:
        # Заглушки для Fast Mode
        features['sampen'] = 0.0
        features['corr_dim'] = 0.0

    return features