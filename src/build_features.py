import os
import pandas as pd
import numpy as np
from src.config import DATA_DIR, DEFAULT_SUBJECTS
from src.data_loader import load_wesad_subject
from src.preprocessing import segment_into_windows
from src.features import extract_features


def main():
    print("Старт генерации признакового пространства...")
    all_features = []
    all_labels = []

    for sub_id in DEFAULT_SUBJECTS:
        file_path = os.path.join(DATA_DIR, f"{sub_id}.pkl")
        if not os.path.exists(file_path):
            print(f"Пропуск {sub_id}: файл не найден")
            continue

        print(f"Обработка субъекта {sub_id}...")
        ecg, labels = load_wesad_subject(file_path)

        df = pd.DataFrame({'ECG': ecg, 'Label': labels})
        df = df[df['Label'].isin([1, 2])].copy()

        rr_windows, y_labels = segment_into_windows(df)
        print(f"  -> Извлечено {len(rr_windows)} окон. Считаем признаки...")

        for i, rr in enumerate(rr_windows):
            feats = extract_features(rr)
            all_features.append(feats)
            all_labels.append(y_labels[i])

    # Формируем итоговый DataFrame
    df_features = pd.DataFrame(all_features)
    df_features['target_label'] = all_labels

    # Очистка от бесконечностей (защита от деления на 0 в LF/HF)
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_features.fillna(0, inplace=True)

    # Сохраняем в CSV
    output_path = os.path.join(DATA_DIR, 'processed_features.csv')
    df_features.to_csv(output_path, index=False)
    print(f"Готово! Датасет сохранен в: {output_path}")
    print(f"Размерность: {df_features.shape}")


if __name__ == "__main__":
    main()