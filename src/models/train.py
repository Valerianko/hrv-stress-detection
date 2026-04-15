import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from src.config import DATA_DIR, MODELS_DIR


def main():
    data_path = os.path.join(DATA_DIR, 'processed_features.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл {data_path} не найден. Сначала запустите src/build_features.py")

    print("Загрузка данных...")
    df = pd.read_csv(data_path)

    X = df.drop(columns=['target_label'])
    y = df['target_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    print("\nИнициализация моделей для сравнения...")
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                        random_state=42)
    }

    best_model = None
    best_auc = 0.0
    best_name = ""

    for name, model in models.items():
        print(f"\n--- Обучение: {name} ---")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name

    print("\n========================================")
    print(f" Лучшая модель: {best_name} (AUC: {best_auc:.4f})")
    print("========================================")
    print(classification_report(y_test, best_model.predict(X_test), target_names=['Покой', 'Стресс']))

    # Сохраняем лучшую модель на диск!
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'best_stress_model.joblib')
    joblib.dump(best_model, model_path)
    print(f" Лучшая модель сохранена в: {model_path}")


if __name__ == "__main__":
    main()