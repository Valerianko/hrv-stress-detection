import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR
from src.data_loader import load_wesad_subject
from src.preprocessing import process_ecg_to_rr
from src.models.inference import StressAnalyzer

st.set_page_config(page_title="HRV Monitor", page_icon="🫀", layout="wide")

# ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
if 'history_prob' not in st.session_state: st.session_state.history_prob = []
if 'history_hr' not in st.session_state: st.session_state.history_hr = []
if 'history_time' not in st.session_state: st.session_state.history_time = []
if 'history_proc_time' not in st.session_state: st.session_state.history_proc_time = []
if 'last_result' not in st.session_state: st.session_state.last_result = None
if 'prev_metrics' not in st.session_state: st.session_state.prev_metrics = None
if 'current_subject' not in st.session_state: st.session_state.current_subject = None


@st.cache_resource
def get_analyzer():
    return StressAnalyzer()


@st.cache_data
def get_data(subject_id):
    ecg, labels = load_wesad_subject(os.path.join(DATA_DIR, f"{subject_id}.pkl"))
    stress_idx = np.where(labels == 2)[0]
    start_idx = max(0, stress_idx[0] - 60 * 700) if len(stress_idx) > 0 else 0
    return ecg, start_idx


def reset_state(start_idx):
    st.session_state.is_running = False
    st.session_state.current_idx = start_idx
    st.session_state.history_prob = []
    st.session_state.history_hr = []
    st.session_state.history_time = []
    st.session_state.history_proc_time = []
    st.session_state.last_result = None
    st.session_state.prev_metrics = None

def plot_attractor(rr, delay=1):
    x, y, z = rr[:-2 * delay], rr[delay:-delay], rr[2 * delay:]
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=4, color=z, colorscale='Viridis', opacity=0.8)
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        title="Фазовый портрет (Аттрактор)",
        height=350,
        scene=dict(xaxis_title='RR(t)', yaxis_title='RR(t+1)', zaxis_title='RR(t+2)')
    )
    return fig


def plot_trend():
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Зона низкого риска (Покой) - Надежный метод через add_shape
    fig.add_shape(
        type="rect",
        x0=0, x1=1, xref="paper",
        y0=0.0, y1=0.5, yref="y",
        fillcolor="rgba(46, 204, 113, 0.25)",  # Изумрудный с 25% непрозрачности
        line_width=0,
        layer="below"
    )

    # Зона высокого риска (Стресс)
    fig.add_shape(
        type="rect",
        x0=0, x1=1, xref="paper",
        y0=0.5, y1=1.05, yref="y",
        fillcolor="rgba(255, 75, 75, 0.25)",  # Красный с 25% непрозрачности
        line_width=0,
        layer="below"
    )

    if st.session_state.history_time:
        # Линия 1: Вероятность стресса
        fig.add_trace(go.Scatter(
            x=st.session_state.history_time,
            y=st.session_state.history_prob,
            mode='lines+markers',
            line=dict(color='#ff4b4b', width=3),
            marker=dict(size=6),
            name="Вероятность стресса"
        ), secondary_y=False)

        # Линия 2: Пульс (ЧСС)
        fig.add_trace(go.Scatter(
            x=st.session_state.history_time,
            y=st.session_state.history_hr,
            mode='lines',
            line=dict(color='#00d4ff', width=2, dash='dot'),
            name="Пульс (уд/мин)",
            opacity=0.85
        ), secondary_y=True)

    # Линия порога
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="white",
        line_width=2,
        secondary_y=False
    )

    fig.update_layout(
        title="Динамика состояния пациента",
        height=350,
        margin=dict(l=0, r=0, b=0, t=30),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Вероятность", range=[0, 1.05], secondary_y=False)
    fig.update_yaxes(title_text="Пульс", secondary_y=True, showgrid=False)

    return fig

st.sidebar.title("⚙️ Настройки")

data_source = st.sidebar.radio("Источник данных:", ["База пациентов (WESAD)", "Загрузить новый файл"])

if data_source == "База пациентов (WESAD)":
    subject = st.sidebar.selectbox("Пациент:",
                                   ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"])
    ecg, start_idx = get_data(subject)
else:
    # ЗАГЛУШКА ДЛЯ НОВЫХ ПАЦИЕНТОВ
    uploaded_file = st.sidebar.file_uploader("Загрузите файл ЭКГ (.csv или .pkl)", type=['csv', 'pkl'])
    if uploaded_file is not None:
        st.sidebar.warning(
            "⚠️  Загрузка пользовательских ЭКГ-файлов предусмотрена в архитектуре системы, но в текущей версии доступна только демонстрация на базе WESAD.")

    # Фиктивные данные для безопасной работы UI
    subject = "Custom_Upload"
    ecg, start_idx = [], 0

# Сброс состояния при смене пациента/источника
if subject != st.session_state.current_subject:
    reset_state(start_idx)
    st.session_state.current_subject = subject

st.sidebar.markdown("---")
mode = st.sidebar.radio("Режим анализа:", ["Fast (Быстрый)", "Full (Полный с аттракторами)"])
speed_sec = st.sidebar.slider("Шаг симуляции (сек):", 5, 30, 10, 5)

is_fast = "Fast" in mode
step_size_sec = speed_sec
fs = 700
window_size = 60 * fs
step_size = step_size_sec * fs

st.sidebar.markdown("---")
st.sidebar.info(
    f"🧠 **Активная модель:**\n\n`{'model_fast.joblib' if is_fast else 'model_full.joblib'}`\n\nАлгоритм: *SVM (RBF)*"
)

if st.sidebar.button("🔄 Сбросить историю", use_container_width=True):
    reset_state(start_idx)

col_btn1, col_btn2 = st.sidebar.columns(2)

# Блокируем кнопку старт, если выбран пользовательский файл (защита от ошибок)
start_disabled = (data_source == "Загрузить новый файл")

if col_btn1.button("▶️ Старт", type="primary", use_container_width=True, disabled=start_disabled):
    st.session_state.is_running = True
if col_btn2.button("⏸ Пауза", use_container_width=True):
    st.session_state.is_running = False

# ЛОГИКА ШАГА СИМУЛЯЦИИ
analyzer = get_analyzer()

if st.session_state.is_running and len(ecg) > 0:
    if st.session_state.current_idx < len(ecg) - window_size:
        segment = ecg[st.session_state.current_idx: st.session_state.current_idx + window_size]
        rr = process_ecg_to_rr(segment, fs=fs)

        if len(rr) > 30:
            res = analyzer.predict_window(rr, fast_mode=is_fast)

            if not res.get("error"):
                # Сохраняем предыдущие метрики для расчета дельты
                st.session_state.prev_metrics = st.session_state.last_result[
                    'features'] if st.session_state.last_result else res['features']
                st.session_state.last_result = res
                st.session_state.last_result['rr'] = rr

                # Обновляем историю
                current_time_sec = (st.session_state.current_idx - start_idx) // fs
                st.session_state.history_prob.append(res['probability'])
                st.session_state.history_hr.append(60000 / res['features']['mean_rr'])
                st.session_state.history_time.append(current_time_sec)
                st.session_state.history_proc_time.append(res['processing_time_sec'])

        # Сдвигаем окно
        st.session_state.current_idx += step_size
    else:
        st.session_state.is_running = False
        st.sidebar.success("Запись завершена.")

st.title("🫀 Интеллектуальная система мониторинга стресса")

if data_source == "Загрузить новый файл":
    st.info("📂 Режим пользовательских данных активен. Пожалуйста, загрузите файл ЭКГ в панели слева.")
elif not st.session_state.last_result:
    st.info("👈 Выберите пациента в меню слева и нажмите 'Старт' для начала симуляции потокового мониторинга.")

if st.session_state.last_result and data_source == "База пациентов (WESAD)":
    res = st.session_state.last_result
    curr_f = res['features']
    prev_f = st.session_state.prev_metrics or curr_f

    if res['status'] == "СТРЕСС":
        st.error(f"⚠️ **{res['status']}** | Вероятность: {res['probability']:.1%} | Анализ: {res['mode']}")
    else:
        st.success(f"🍃 **{res['status']}** | Вероятность: {res['probability']:.1%} | Анализ: {res['mode']}")

    # Метрики
    c1, c2, c3, c4 = st.columns(4)
    pulse = 60000 / curr_f['mean_rr']
    prev_pulse = 60000 / prev_f['mean_rr'] if prev_f else pulse

    c1.metric("Пульс (уд/мин)", f"{pulse:.0f}", f"{pulse - prev_pulse:.1f}", delta_color="inverse")
    c2.metric("SDNN (мс)", f"{curr_f['sdnn']:.1f}", f"{curr_f['sdnn'] - prev_f.get('sdnn', curr_f['sdnn']):.1f}")
    c3.metric("LF/HF (Баланс)", f"{curr_f['lf_hf_ratio']:.2f}",
              f"{curr_f['lf_hf_ratio'] - prev_f.get('lf_hf_ratio', curr_f['lf_hf_ratio']):.2f}", delta_color="inverse")

    if not is_fast:
        c4.metric("Corr Dim (D2)", f"{curr_f['corr_dim']:.2f}",
                  f"{curr_f['corr_dim'] - prev_f.get('corr_dim', curr_f['corr_dim']):.2f}")
    else:
        c4.metric("Corr Dim (D2)", "Отключено", delta_color="off",
                  help="Нелинейные признаки отключены в быстром режиме")

    st.markdown("---")

    # Графики
    col_plot1, col_plot2 = st.columns([1, 1.2])

    with col_plot1:
        if not is_fast:
            st.plotly_chart(plot_attractor(res['rr']), use_container_width=True)
        else:
            st.info("ℹ️ Построение аттракторов отключено в режиме Fast Mode для минимизации задержек (Latency).")

    with col_plot2:
        st.plotly_chart(plot_trend(), use_container_width=True)

        # Вывод статистики производительности
        if st.session_state.history_proc_time:
            avg_time = np.mean(st.session_state.history_proc_time)
            margin = step_size_sec - avg_time
            st.caption(
                f"⏱️ Среднее время обработки окна: **{avg_time:.3f} сек.** | Запас до real-time: **{margin:.3f} сек.**")


if st.session_state.is_running:
    time.sleep(0.8)
    st.rerun()