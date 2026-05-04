import os
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from src.config import DATA_DIR, MODELS_DIR, FIGURES_DIR
from src.logger import get_logger

logger = get_logger("SHAP_Explainer")


def main():
    logger.info("Запуск модуля интерпретации SHAP...")

    # Загрузка данных и модели
    data_path = os.path.join(DATA_DIR, 'processed_features.csv')
    model_path = os.path.join(MODELS_DIR, 'best_stress_model.joblib')

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        logger.error("Нет данных или модели. Сначала запустите train.py")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=['target_label', 'subject_id'])

    model = joblib.load(model_path)
    logger.info("Данные и модель (SVM) успешно загружены.")

    # 2. Настройка SHAP для SVM (KernelExplainer)
    # Так как KernelExplainer медленный, берем выборку K-Means (100 типичных окон)
    # в качестве базового фона (background)
    logger.info("Подготовка фонового датасета (K-Means) для ускорения SHAP...")
    background = shap.kmeans(X, 100)

    # Функция предсказания вероятностей (нужно для SHAP)
    predict_proba_func = lambda x: model.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(predict_proba_func, background)

    # Берем 200 случайных окон для построения красивого графика
    logger.info("Расчет SHAP Values для 200 окон (это может занять пару минут)...")
    X_sample = X.sample(n=200, random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # Визуализация и сохранение (Summary Plot)
    logger.info("Генерация SHAP Summary Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)

    # Локальная интерпретация (Local Explanation) для одного окна со стрессом
    logger.info("Генерация локального объяснения (Waterfall Plot)...")

    stress_indices = df[df['target_label'] == 1].index
    if len(stress_indices) > 0:
        sample_idx = stress_indices[0]  # Берем первое окно стресса
        instance = X.iloc[sample_idx]

        # Для Waterfall нам нужен объект Explanation
        shap_val_single = explainer.shap_values(instance)

        # Создаем объект Explanation вручную
        explanation = shap.Explanation(
            values=shap_val_single,
            base_values=explainer.expected_value,
            data=instance.values,
            feature_names=X.columns
        )

        # Строим и сохраняем Waterfall plot
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)

        fig_path_local = os.path.join(FIGURES_DIR, 'shap_local_explanation.png')
        plt.tight_layout()
        plt.savefig(fig_path_local, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Локальное объяснение сохранено в: {fig_path_local}")
    else:
        logger.warning("Окна со стрессом не найдены, локальный график не построен.")


if __name__ == "__main__":
    main()