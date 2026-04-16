import os
import joblib
import pandas as pd
import numpy as np
import time
from typing import Dict, Union
from src.config import MODELS_DIR
from src.features import extract_features
from src.logger import get_logger

logger = get_logger("Inference")


class StressAnalyzer:
    def __init__(self):
        # Инициализация анализатора: загрузка модели в память
        model_path = os.path.join(MODELS_DIR, 'best_stress_model.joblib')
        if not os.path.exists(model_path):
            logger.error(f"Модель не найдена по пути: {model_path}")
            raise FileNotFoundError("Модель не обучена.")

        self.model = joblib.load(model_path)
        logger.info("StressAnalyzer успешно инициализирован (модель загружена).")

    def predict_window(self, rr_intervals: np.ndarray, fast_mode: bool = False) -> Dict[str, Union[str, float, dict]]:
        """
        Полный цикл анализа одного окна RR-интервалов.

        Args:
            rr_intervals: Массив RR-интервалов (мс).
            fast_mode: Если True, нелинейные метрики (Takens) не считаются (для экономии CPU).

        Returns:
            Словарь с результатами: статус, вероятность, время обработки, извлеченные признаки.
        """
        start_time = time.time()

        try:
            # Извлечение признаков
            features = extract_features(rr_intervals, calc_nonlinear=not fast_mode)

            # Подготовка вектора
            df_feat = pd.DataFrame([features])
            df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_feat.fillna(0, inplace=True)

            # Инференс
            pred = self.model.predict(df_feat)[0]
            prob = self.model.predict_proba(df_feat)[0][pred]

            status = "СТРЕСС" if pred == 1 else "ПОКОЙ"

            # Если fast_mode, модель (обученная на 10 признаках) всё равно отработает,
            # так как признаки sampen и corr_dim будут заполнены нулями.
            # Это может слегка снизить точность в моменте, но ускорит расчет.

            processing_time = time.time() - start_time

            result = {
                "status": status,
                "probability": float(prob),
                "processing_time_sec": round(processing_time, 3),
                "features": features,
                "mode": "Fast" if fast_mode else "Full",
                "error": None
            }
            return result

        except Exception as e:
            logger.error(f"Ошибка при анализе окна: {e}")
            return {"error": str(e)}


# Пример использования (для тестов)
if __name__ == "__main__":
    analyzer = StressAnalyzer()
    dummy_rr = np.random.normal(800, 50, 70)  # Симуляция 70 ударов (покой)

    print("\n--- ТЕСТ FULL MODE ---")
    res_full = analyzer.predict_window(dummy_rr, fast_mode=False)
    print(res_full)

    print("\n--- ТЕСТ FAST MODE ---")
    res_fast = analyzer.predict_window(dummy_rr, fast_mode=True)
    print(res_fast)