import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR
from src.logger import get_logger

logger = get_logger("Train")


def main():
    data_path = os.path.join(DATA_DIR, 'processed_features.csv')
    df = pd.read_csv(data_path)
    logger.info(f"Данные загружены. Размер: {df.shape}")

    X = df.drop(columns=['target_label', 'subject_id'])
    y = df['target_label'].values
    groups = df['subject_id'].values

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                        random_state=42),
        "SVM (RBF)": Pipeline(
            [('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', probability=True, random_state=42))])
    }

    logo = LeaveOneGroupOut()
    results = []

    logger.info("Старт LOSO Cross-Validation...")

    # Для сохранения данных лучшей модели для графиков
    best_y_true, best_y_prob, best_y_pred = [], [], []
    best_auc_global = 0.0
    best_model_name = ""

    for name, template in models.items():
        logger.info(f"Обучение {name}...")
        fold_acc, fold_f1, fold_auc = [], [], []

        y_true_all, y_prob_all, y_pred_all = [], [], []

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = clone(template)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            y_true_all.extend(y_test)
            y_prob_all.extend(y_prob)
            y_pred_all.extend(y_pred)

            fold_acc.append(accuracy_score(y_test, y_pred))
            fold_f1.append(f1_score(y_test, y_pred, zero_division=0))
            if len(np.unique(y_test)) > 1:
                fold_auc.append(roc_auc_score(y_test, y_prob))

        mean_auc = np.mean(fold_auc) if fold_auc else 0.0
        results.append({
            "Model": name,
            "Accuracy": np.mean(fold_acc),
            "F1-score": np.mean(fold_f1),
            "ROC AUC": mean_auc
        })

        if mean_auc > best_auc_global:
            best_auc_global = mean_auc
            best_model_name = name
            best_y_true, best_y_prob, best_y_pred = y_true_all, y_prob_all, y_pred_all

    # Логируем результаты
    df_results = pd.DataFrame(results).sort_values(by="ROC AUC", ascending=False)
    logger.info(f"\n{df_results.to_string(index=False)}")
    df_results.to_csv(os.path.join(REPORTS_DIR, 'tables', 'model_comparison.csv'), index=False)

    logger.info("Генерация графиков...")

    # ROC Curve
    fpr, tpr, _ = roc_curve(best_y_true, best_y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{best_model_name} (AUC = {best_auc_global:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая (LOSO Validation)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curve.png'), bbox_inches='tight')
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(best_y_true, best_y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Покой', 'Стресс'], yticklabels=['Покой', 'Стресс'])
    plt.title(f'Матрица ошибок ({best_model_name})')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    # ОБУЧЕНИЕ ДВУХ PRODUCTION МОДЕЛЕЙ
    logger.info(f"Обучение Production-моделей на базе {best_model_name}...")
    best_template = models[best_model_name]

    model_full = clone(best_template)
    model_full.fit(X, y)
    joblib.dump(model_full, os.path.join(MODELS_DIR, 'model_full.joblib'))

    X_fast = X.drop(columns=['sampen', 'corr_dim'])
    model_fast = clone(best_template)
    model_fast.fit(X_fast, y)
    joblib.dump(model_fast, os.path.join(MODELS_DIR, 'model_fast.joblib'))

    logger.info("Обучение завершено. Графики и модели сохранены.")


if __name__ == "__main__":
    main()