import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go

# Добавляем корневую папку в sys.path, чтобы Streamlit видел папку src
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, MODELS_DIR
from src.data_loader import load_wesad_subject
from src.preprocessing import process_ecg_to_rr
from src.features import extract_features

# Настройка страницы
st.set_page_config(page_title="HRV Stress Monitor", page_icon="🫀", layout="wide")

# 1. ЗАГРУЗКА РЕСУРСОВ (КЭШИРОВАНИЕ)
@st.cache_resource
def load_model():
    model_path = os.path.join(MODELS_DIR, 'best_stress_model.joblib')
    return joblib.load(model_path)


@st.cache_data
def get_subject_data(subject_id):
    file_path = os.path.join(DATA_DIR, f"{subject_id}.pkl")
    ecg, labels = load_wesad_subject(file_path)
    return ecg, labels


def plot_3d_attractor_plotly(rr_intervals, delay=1):
    # Строит интерактивный 3D аттрактор
    x = rr_intervals[:-2 * delay]
    y = rr_intervals[delay:-delay]
    z = rr_intervals[2 * delay:]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=4, color=z, colorscale='Viridis', opacity=0.8)
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        title="Фазовый портрет (Теорема Такенса)",
        scene=dict(xaxis_title='RR(t)', yaxis_title=f'RR(t+{delay})', zaxis_title=f'RR(t+{2 * delay})')
    )
    return fig


# 2. ИНТЕРФЕЙС ПРИЛОЖЕНИЯ
st.title("🫀 Система мультимодального мониторинга стресса")
st.markdown("Система анализирует вариабельность сердечного ритма с использованием методов нелинейной динамики.")

# Боковая панель
st.sidebar.header("⚙Настройки симуляции")
subject = st.sidebar.selectbox("Выберите профиль пациента:", ["S2", "S3", "S4", "S5"])
speed = st.sidebar.slider("Шаг симуляции (сек):", min_value=5, max_value=30, value=10, step=5)

model = load_model()

if st.sidebar.button("Запустить мониторинг", type="primary"):
    st.sidebar.success("Система запущена!")

    # Загружаем данные пациента
    ecg_full, labels_full = get_subject_data(subject)
    fs = 700
    window_size = 60 * fs  # 60 секунд
    step_size = speed * fs

    # Ищем момент, где начинается стресс, чтобы показать интересное
    stress_idx = np.where(labels_full == 2)[0]
    if len(stress_idx) > 0:
        # Начинаем за 3 минуты до стресса, чтобы увидеть переход от Покоя к Стрессу
        start_sim = max(0, stress_idx[0] - 3 * 60 * fs)
    else:
        start_sim = 0

    # Создаем пустые контейнеры для обновления интерфейса
    status_placeholder = st.empty()     # Коробка для статуса (Красный/Зеленый)
    metrics_placeholder = st.empty()    # ОДНА коробка для всех 4-х цифр
    plot_placeholder = st.empty()       # Коробка для 3D графика

    # Симуляция потока
    for i in range(start_sim, len(ecg_full) - window_size, step_size):
        # 1. Берем окно 60 секунд
        segment = ecg_full[i: i + window_size]

        # 2. Предобработка
        try:
            rr = process_ecg_to_rr(segment, fs=fs)
            if len(rr) < 30:
                continue

            # 3. Извлечение признаков
            features = extract_features(rr)
            df_feat = pd.DataFrame([features]).fillna(0)

            # 4. Классификация
            pred = model.predict(df_feat)[0]
            prob = model.predict_proba(df_feat)[0][pred]

            # 5. ОБНОВЛЕНИЕ UI
            # Статус
            if pred == 1:
                status_placeholder.error(f"⚠️ ОБНАРУЖЕН СТРЕСС (Уверенность: {prob:.1%})")
            else:
                status_placeholder.success(f"🍃 СОСТОЯНИЕ ПОКОЯ (Уверенность: {prob:.1%})")

                # 2. Обновляем метрики на одном месте!
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Пульс (уд/мин)", f"{60000 / features['mean_rr']:.0f}")
                col2.metric("Вариабельность (SDNN)", f"{features['sdnn']:.1f} мс")
                col3.metric("Индекс стресса (LF/HF)", f"{features['lf_hf_ratio']:.2f}")
                col4.metric("Сложность аттрактора (D2)", f"{features['corr_dim']:.2f}")

                # 3. Обновляем график на одном месте
            fig = plot_3d_attractor_plotly(rr)
            plot_placeholder.plotly_chart(fig, use_container_width=True)

            # Имитация задержки реального времени
            time.sleep(1.5)

        except Exception as e:
            # Выводим ошибку над графиком, но не ломаем цикл
            status_placeholder.warning(f"Артефакты в сигнале. Калибровка...")
            time.sleep(1)