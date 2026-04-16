import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from src.config import DATA_DIR, MODELS_DIR

REPORTS_TABLES_DIR = os.path.join(os.path.dirname(MODELS_DIR), 'reports', 'tables')
os.makedirs(REPORTS_TABLES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    data_path = os.path.join(DATA_DIR, 'processed_features.csv')
    df = pd.read_csv(data_path)
    print(f"Данные загружены. Размер: {df.shape}")

    X = df.drop(columns=['target_label', 'subject_id'])
    y = df['target_label'].values
    groups = df['subject_id'].values

    # Фабрика моделей
    model_templates = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                        random_state=42),
        "SVM (RBF)": Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', probability=True, random_state=42))
        ])
    }

    logo = LeaveOneGroupOut()
    results = []

    print("\nСтарт LOSO Cross-Validation...")

    for name, template in model_templates.items():
        print(f"Обучение {name}...")
        fold_acc, fold_f1, fold_auc = [], [], []

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # КЛОНИРУЕМ модель для чистоты каждого фолда
            model = clone(template)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            fold_acc.append(accuracy_score(y_test, y_pred))
            fold_f1.append(f1_score(y_test, y_pred, zero_division=0))

            # ЗАЩИТА: Считаем AUC только если в тесте есть оба класса
            if len(np.unique(y_test)) > 1:
                fold_auc.append(roc_auc_score(y_test, y_prob))

        results.append({
            "Model": name,
            "Accuracy (Mean)": np.mean(fold_acc),
            "F1-score (Mean)": np.mean(fold_f1),
            "ROC AUC (Mean)": np.mean(fold_auc) if fold_auc else 0.0
        })

    df_results = pd.DataFrame(results).sort_values(by="ROC AUC (Mean)", ascending=False)

    print("\nИтоговое сравнение моделей (LOSO Validation):")
    print(df_results.to_string(index=False))

    report_path = os.path.join(REPORTS_TABLES_DIR, 'model_comparison.csv')
    df_results.to_csv(report_path, index=False)

    best_model_name = df_results.iloc[0]["Model"]
    print(f"\nЛучшая модель: {best_model_name}. Обучаем Production-версию...")

    # Обучаем финальную модель
    best_model = clone(model_templates[best_model_name])
    best_model.fit(X, y)

    model_path = os.path.join(MODELS_DIR, 'best_stress_model.joblib')
    joblib.dump(best_model, model_path)


if __name__ == "__main__":
    main()