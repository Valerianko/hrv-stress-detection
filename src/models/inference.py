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
        """Инициализация анализатора: загрузка двух моделей."""
        try:
            self.model_full = joblib.load(os.path.join(MODELS_DIR, 'model_full.joblib'))
            self.model_fast = joblib.load(os.path.join(MODELS_DIR, 'model_fast.joblib'))
            logger.info("Модели Full и Fast успешно загружены.")
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            raise

    def predict_window(self, rr_intervals: np.ndarray, fast_mode: bool = False) -> Dict[str, Union[str, float, dict]]:
        start_time = time.time()

        try:
            features = extract_features(rr_intervals, calc_nonlinear=not fast_mode)
            df_feat = pd.DataFrame([features])

            if fast_mode:
                df_feat = df_feat.drop(columns=['sampen', 'corr_dim'], errors='ignore')
                model = self.model_fast
            else:
                model = self.model_full

            df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_feat.fillna(0, inplace=True)

            # Истинная вероятность (Вероятность класса 1 - СТРЕСС)
            proba_array = model.predict_proba(df_feat)[0]
            stress_prob = float(proba_array[1])  # Берем именно вероятность стресса

            # Вердикт строго по порогу 0.5
            pred_class = 1 if stress_prob >= 0.5 else 0
            status = "СТРЕСС" if pred_class == 1 else "ПОКОЙ"

            proc_time = time.time() - start_time

            return {
                "status": status,
                "probability": stress_prob,
                "processing_time_sec": round(proc_time, 3),
                "features": features,
                "mode": "Fast" if fast_mode else "Full",
                "error": None
            }
        except Exception as e:
            logger.error(f"Ошибка инференса (Fast: {fast_mode}, Удары: {len(rr_intervals)}): {str(e)}")
            return {"error": str(e)}