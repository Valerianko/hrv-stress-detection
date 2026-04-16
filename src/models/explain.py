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

    fig_path = os.path.join(FIGURES_DIR, 'shap_summary_svm.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"График SHAP успешно сохранен в: {fig_path}")


if __name__ == "__main__":
    main()