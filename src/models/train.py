import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, \
    average_precision_score, PrecisionRecallDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR
from src.logger import get_logger

logger = get_logger("Train")
REPORTS_TABLES_DIR = os.path.join(REPORTS_DIR, 'tables')
os.makedirs(REPORTS_TABLES_DIR, exist_ok=True)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'processed_features.csv'))
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
    subject_results = []  # Для отчета по каждому пациенту

    logger.info("Старт LOSO Cross-Validation...")

    best_y_true, best_y_prob, best_y_pred = [], [], []
    best_auc_global = 0.0
    best_model_name = ""

    for name, template in models.items():
        fold_acc, fold_f1, fold_auc, fold_pr = [], [], [], []
        y_true_all, y_prob_all, y_pred_all = [], [], []

        for train_idx, test_idx in logo.split(X, y, groups):
            current_subject = groups[test_idx[0]]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = clone(template)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            y_true_all.extend(y_test)
            y_prob_all.extend(y_prob)
            y_pred_all.extend(y_pred)

            # Метрики фолда
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            fold_acc.append(acc)
            fold_f1.append(f1)

            auc_val, pr_auc = 0.0, 0.0
            if len(np.unique(y_test)) > 1:
                auc_val = roc_auc_score(y_test, y_prob)
                pr_auc = average_precision_score(y_test, y_prob)
                fold_auc.append(auc_val)
                fold_pr.append(pr_auc)

            # Сохраняем результат для конкретного человека (только для лучшей модели в будущем)
            subject_results.append(
                {"Model": name, "Subject": current_subject, "Accuracy": acc, "F1": f1, "AUC": auc_val})

        mean_auc = np.mean(fold_auc) if fold_auc else 0.0
        results.append({
            "Model": name,
            "Accuracy": np.mean(fold_acc),
            "F1-score": np.mean(fold_f1),
            "ROC AUC": mean_auc,
            "PR AUC": np.mean(fold_pr) if fold_pr else 0.0
        })

        if mean_auc > best_auc_global:
            best_auc_global = mean_auc
            best_model_name = name
            best_y_true, best_y_prob, best_y_pred = y_true_all, y_prob_all, y_pred_all

    # Сохраняем сводную таблицу
    df_results = pd.DataFrame(results).sort_values(by="ROC AUC", ascending=False)
    df_results.to_csv(os.path.join(REPORTS_TABLES_DIR, 'model_comparison.csv'), index=False)

    # Сохраняем отчет по субъектам для ЛУЧШЕЙ модели
    df_subj = pd.DataFrame(subject_results)
    df_subj_best = df_subj[df_subj["Model"] == best_model_name].sort_values(by="AUC", ascending=False)
    df_subj_best.to_csv(os.path.join(REPORTS_TABLES_DIR, 'per_subject_performance.csv'), index=False)

    logger.info(f"\nЛучшая модель: {best_model_name} (AUC: {best_auc_global:.4f})")
    logger.info(
        f"Лучший субъект: {df_subj_best.iloc[0]['Subject']} | Худший субъект: {df_subj_best.iloc[-1]['Subject']}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(best_y_true, best_y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {best_auc_global:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'ROC-кривая ({best_model_name})')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curve.png'), bbox_inches='tight')
    plt.close()

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(best_y_true, best_y_prob, name=best_model_name)
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(FIGURES_DIR, 'pr_curve.png'), bbox_inches='tight')
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(best_y_true, best_y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Матрица ошибок (LOSO)')
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    # ОБУЧЕНИЕ FULL и FAST версий
    logger.info("Обучение Production-моделей...")
    best_template = models[best_model_name]

    # FULL
    model_full = clone(best_template).fit(X, y)
    joblib.dump(model_full, os.path.join(MODELS_DIR, 'model_full.joblib'))

    # FAST
    X_fast = X.drop(columns=['sampen', 'corr_dim'])
    model_fast = clone(best_template).fit(X_fast, y)
    joblib.dump(model_fast, os.path.join(MODELS_DIR, 'model_fast.joblib'))

    # Оценка FAST на всем датасете
    y_fast_pred = model_fast.predict(X_fast)
    fast_acc = accuracy_score(y, y_fast_pred)
    full_acc = accuracy_score(y, model_full.predict(X))
    logger.info(f"Сравнение: Точность Full-режима = {full_acc:.4f} | Точность Fast-режима = {fast_acc:.4f}")


if __name__ == "__main__":
    main()