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

# ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ (STATE MACHINE)
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
if 'history_prob' not in st.session_state: st.session_state.history_prob = []
if 'history_time' not in st.session_state: st.session_state.history_time = []
if 'last_result' not in st.session_state: st.session_state.last_result = None
if 'prev_metrics' not in st.session_state: st.session_state.prev_metrics = None


@st.cache_resource
def get_analyzer():
    return StressAnalyzer()


@st.cache_data
def get_data(subject_id):
    ecg, labels = load_wesad_subject(os.path.join(DATA_DIR, f"{subject_id}.pkl"))
    # Находим индекс старта (за 3 мин до первого стресса, если он есть)
    stress_idx = np.where(labels == 2)[0]
    start_idx = max(0, stress_idx[0] - 60 * 700) if len(stress_idx) > 0 else 0
    return ecg, start_idx


def reset_state(start_idx):
    st.session_state.is_running = False
    st.session_state.current_idx = start_idx
    st.session_state.history_prob = []
    st.session_state.history_time = []
    st.session_state.last_result = None
    st.session_state.prev_metrics = None


def plot_attractor(rr, delay=1):
    x, y, z = rr[:-2 * delay], rr[delay:-delay], rr[2 * delay:]
    fig = go.Figure(
        data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3, color=z, colorscale='Viridis'))])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), title="Аттрактор", height=400)
    return fig


def plot_trend():
    fig = go.Figure()
    if st.session_state.history_time:
        fig.add_trace(go.Scatter(x=st.session_state.history_time, y=st.session_state.history_prob,
                                 mode='lines+markers', line=dict(color='red', width=3)))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Порог стресса")
    fig.update_layout(title="История вероятности стресса", yaxis_title="Вероятность", yaxis=dict(range=[0, 1]),
                      height=300, margin=dict(l=0, r=0, b=0, t=30))
    return fig


st.sidebar.title("Настройки")
subject = st.sidebar.selectbox("Пациент:", ["S2", "S3", "S4", "S5", "S6"])
mode = st.sidebar.radio("Режим анализа:", ["Fast (Быстрый)", "Full (Полный)"])
speed_sec = st.sidebar.slider("Шаг симуляции (сек):", 5, 30, 10, 5)

ecg, start_idx = get_data(subject)

# Если сменили пациента или нажали сброс — обнуляем состояние
if st.sidebar.button("Сбросить историю", use_container_width=True) or st.session_state.current_idx == 0:
    reset_state(start_idx)

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("▶️ Старт", use_container_width=True):
    st.session_state.is_running = True
if col_btn2.button("⏸ Пауза", use_container_width=True):
    st.session_state.is_running = False


# ЛОГИКА ШАГА СИМУЛЯЦИИ
analyzer = get_analyzer()
fs = 700
window_size = 60 * fs
step_size = speed_sec * fs
is_fast = "Fast" in mode

if st.session_state.is_running:
    if st.session_state.current_idx < len(ecg) - window_size:
        segment = ecg[st.session_state.current_idx: st.session_state.current_idx + window_size]
        rr = process_ecg_to_rr(segment, fs=fs)

        if len(rr) > 30:
            res = analyzer.predict_window(rr, fast_mode=is_fast)
            if not res.get("error"):
                # Сохраняем результат в State
                st.session_state.prev_metrics = st.session_state.last_result[
                    'features'] if st.session_state.last_result else res['features']
                st.session_state.last_result = res
                st.session_state.last_result['rr'] = rr  # Сохраняем RR для графика

                # Обновляем историю тренда
                current_time_sec = (st.session_state.current_idx - start_idx) // fs
                st.session_state.history_prob.append(res['probability'])
                st.session_state.history_time.append(current_time_sec)

        # Сдвигаем окно вперед
        st.session_state.current_idx += step_size
    else:
        st.session_state.is_running = False
        st.sidebar.success("Запись завершена.")

st.title("🫀 Интеллектуальная система мониторинга стресса")

if st.session_state.last_result:
    res = st.session_state.last_result
    curr_f = res['features']
    prev_f = st.session_state.prev_metrics

    if res['status'] == "СТРЕСС":
        st.error(
            f"⚠️ {res['status']} | Вероятность: {res['probability']:.1%} | Время расчета: {res['processing_time_sec']}с")
    else:
        st.success(
            f"🍃 {res['status']} | Вероятность: {res['probability']:.1%} | Время расчета: {res['processing_time_sec']}с")

    # 2. Метрики с дельтами
    c1, c2, c3, c4 = st.columns(4)
    pulse, prev_pulse = 60000 / curr_f['mean_rr'], 60000 / prev_f['mean_rr']
    c1.metric("Пульс", f"{pulse:.0f}", f"{pulse - prev_pulse:.1f}", delta_color="inverse")
    c2.metric("SDNN (мс)", f"{curr_f['sdnn']:.1f}", f"{curr_f['sdnn'] - prev_f['sdnn']:.1f}")
    c3.metric("LF/HF", f"{curr_f['lf_hf_ratio']:.2f}", f"{curr_f['lf_hf_ratio'] - prev_f['lf_hf_ratio']:.2f}",
              delta_color="inverse")

    if not is_fast:
        c4.metric("Corr Dim (D2)", f"{curr_f['corr_dim']:.2f}", f"{curr_f['corr_dim'] - prev_f['corr_dim']:.2f}")
    else:
        c4.metric("Corr Dim (D2)", "N/A (Fast Mode)", delta_color="off")

    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        if not is_fast:
            st.plotly_chart(plot_attractor(res['rr']), use_container_width=True)
        else:
            st.info("Аттрактор отключен в режиме Fast Mode для экономии ресурсов.")
    with col_plot2:
        st.plotly_chart(plot_trend(), use_container_width=True)

else:
    st.info("Нажмите 'Старт', чтобы начать мониторинг пациента.")


if st.session_state.is_running:
    time.sleep(1)
    st.rerun()