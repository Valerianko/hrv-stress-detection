
import os
import pandas as pd
from src.config import DATA_DIR
from src.data_loader import load_wesad_subject
from src.preprocessing import segment_into_windows
from src.features import extract_features

if __name__ == "__main__":
    file_path = os.path.join(DATA_DIR, 'S2.pkl')

    print("1. Загрузка...")
    ecg, labels = load_wesad_subject(file_path)

    print("2. Фильтрация классов...")
    df = pd.DataFrame({'ECG': ecg, 'Label': labels})
    df = df[df['Label'].isin([1, 2])].copy()

    print("3. Нарезка на окна...")
    rr_windows, y_labels = segment_into_windows(df)
    print(f"Получено {len(rr_windows)} окон.")

    print("4. Извлечение признаков для первого окна...")
    features = extract_features(rr_windows[0])
    print("Признаки:", features)