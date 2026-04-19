import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import List, Tuple
from src.config import SAMPLING_RATE, WINDOW_SIZE_SEC, STEP_SIZE_SEC, MIN_RR_PER_WINDOW


def process_ecg_to_rr(ecg_signal: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:

    # Очищает ЭКГ-сигнал и извлекает последовательность RR-интервалов (в мс).
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="neurokit")
    # Поиск R-пиков
    _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
    r_peaks = info["ECG_R_Peaks"]

    # Расчет RR-интервалов (перевод из отсчетов в миллисекунды)
    rr_intervals = np.diff(r_peaks) / fs * 1000

    return rr_intervals


def segment_into_windows(df_data: pd.DataFrame,
                         window_sec: int = WINDOW_SIZE_SEC,
                         step_sec: int = STEP_SIZE_SEC,
                         fs: int = SAMPLING_RATE) -> Tuple[List[np.ndarray], List[int]]:

    # Разрезает датафрейм (ЭКГ + Метки) на скользящие окна
    # Возвращает список массивов RR-интервалов и список бинарных меток (0 - Покой, 1 - Стресс)

    windows_rr = []
    windows_labels = []

    window_len = window_sec * fs
    step = step_sec * fs

    for i in range(0, len(df_data) - window_len, step):
        segment = df_data.iloc[i: i + window_len]

        # Берем окно, только если в нем 100% времени длится одно состояние
        unique_labels = segment['Label'].unique()

        if len(unique_labels) == 1:
            current_label = unique_labels[0]

            try:
                # Извлекаем RR-интервалы для данного окна
                rr = process_ecg_to_rr(segment['ECG'].values, fs=fs)

                # Защита от сильного шума (если алгоритм не нашел пиков)
                if len(rr) > MIN_RR_PER_WINDOW:
                    windows_rr.append(rr)
                    # Бинаризация: 1 (Baseline) -> 0, 2 (Stress) -> 1
                    ml_label = 0 if current_label == 1 else 1
                    windows_labels.append(ml_label)
            except Exception:
                continue

    return windows_rr, windows_labels