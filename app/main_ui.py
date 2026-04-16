# app/main_ui.py
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR
from src.data_loader import load_wesad_subject
from src.preprocessing import process_ecg_to_rr
from src.models.inference import StressAnalyzer

st.set_page_config(page_title="HRV Monitor", page_icon="🫀", layout="wide")

# Инициализация состояния
if 'history_prob' not in st.session_state:
    st.session_state.history_prob = []
if 'history_time' not in st.session_state:
    st.session_state.history_time = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'prev_metrics' not in st.session_state:
    st.session_state.prev_metrics = None


@st.cache_resource
def get_analyzer():
    return StressAnalyzer()


@st.cache_data
def get_data(subject_id):
    return load_wesad_subject(os.path.join(DATA_DIR, f"{subject_id}.pkl"))


def plot_attractor(rr, delay=1):
    x, y, z = rr[:-2 * delay], rr[delay:-delay], rr[2 * delay:]
    fig = go.Figure(
        data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3, color=z, colorscale='Viridis'))])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), title="Аттрактор", height=400)
    return fig


def plot_trend():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.history_time, y=st.session_state.history_prob,
                             mode='lines+markers', line=dict(color='red', width=3)))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Порог стресса")
    fig.update_layout(title="История вероятности стресса", yaxis_title="Вероятность", yaxis=dict(range=[0, 1]),
                      height=300, margin=dict(l=0, r=0, b=0, t=30))
    return fig


# UI ЛЕЙАУТ
st.title("🫀 Интеллектуальная система мониторинга стресса")

with st.sidebar:
    st.header("Настройки")
    subject = st.selectbox("Пациент:", ["S2", "S3", "S4", "S5", "S6"])
    mode = st.radio("Режим анализа:", ["Fast (Быстрый)", "Full (Полный с аттракторами)"])

    col1, col2 = st.columns(2)
    if col1.button("▶️ Старт", use_container_width=True):
        st.session_state.is_running = True
    if col2.button("⏸ Пауза", use_container_width=True):
        st.session_state.is_running = False

    if st.button("🔄 Сбросить историю", use_container_width=True):
        st.session_state.history_prob = []
        st.session_state.history_time = []
        st.session_state.prev_metrics = None

analyzer = get_analyzer()

# Контейнеры
status_ph = st.empty()
metrics_ph = st.empty()
col_plot1, col_plot2 = st.columns(2)
plot_attractor_ph = col_plot1.empty()
plot_trend_ph = col_plot2.empty()

if st.session_state.is_running:
    ecg, labels = get_data(subject)
    fs, window_size, step_size = 700, 60 * 700, 5 * 700  # Шаг 5 секунд

    stress_idx = np.where(labels == 2)[0]
    start_idx = max(0, stress_idx[0] - 60 * fs) if len(stress_idx) > 0 else 0

    is_fast = "Fast" in mode

    # Начинаем с того места, где остановились (определяем по длине истории)
    current_step = len(st.session_state.history_prob)
    i = start_idx + (current_step * step_size)

    if i < len(ecg) - window_size:
        rr = process_ecg_to_rr(ecg[i: i + window_size])
        if len(rr) > 30:
            res = analyzer.predict_window(rr, fast_mode=is_fast)

            if not res.get("error"):
                # Обновляем историю
                st.session_state.history_prob.append(res['probability'])
                st.session_state.history_time.append(current_step * 5)  # время в секундах

                # Статус
                if res['status'] == "СТРЕСС":
                    status_ph.error(
                        f"⚠️ СТРЕСС | Вероятность: {res['probability']:.1%} | Время расчета: {res['processing_time_sec']}с")
                else:
                    status_ph.success(
                        f"🍃 ПОКОЙ | Вероятность: {res['probability']:.1%} | Время расчета: {res['processing_time_sec']}с")

                # Метрики с дельтами
                curr_f = res['features']
                prev_f = st.session_state.prev_metrics or curr_f

                with metrics_ph.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Пульс", f"{60000 / curr_f['mean_rr']:.0f}",
                              f"{60000 / curr_f['mean_rr'] - 60000 / prev_f['mean_rr']:.1f}", delta_color="inverse")
                    c2.metric("SDNN (мс)", f"{curr_f['sdnn']:.1f}", f"{curr_f['sdnn'] - prev_f['sdnn']:.1f}")
                    c3.metric("LF/HF", f"{curr_f['lf_hf_ratio']:.2f}",
                              f"{curr_f['lf_hf_ratio'] - prev_f['lf_hf_ratio']:.2f}", delta_color="inverse")
                    c4.metric("Corr Dim (D2)", f"{curr_f['corr_dim']:.2f}" if not is_fast else "N/A")

                st.session_state.prev_metrics = curr_f

                # Графики
                if not is_fast:
                    plot_attractor_ph.plotly_chart(plot_attractor(rr), use_container_width=True)
                else:
                    plot_attractor_ph.info("Аттрактор отключен в режиме Fast Mode")

                plot_trend_ph.plotly_chart(plot_trend(), use_container_width=True)

        time.sleep(1)  # Задержка для визуализации
        st.rerun()  # Перезапуск скрипта для следующего шага
    else:
        st.success("Симуляция завершена (конец записи).")
        st.session_state.is_running = False