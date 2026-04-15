import pickle
import numpy as np
from typing import Tuple

def load_wesad_subject(subject_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает данные субъекта из .pkl файла датасета WESAD.

    Args:
        subject_path (str): Локальный путь к файлу (например, 'data/S2/S2.pkl')

    Returns:
        Tuple[np.ndarray, np.ndarray]: Сырой сигнал ЭКГ (1D массив) и массив меток классов.
    """
    with open(subject_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    # Извлекаем сигнал ЭКГ
    ecg_signal = data['signal']['chest']['ECG'].flatten()

    labels = data['label']

    return ecg_signal, labels