import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models_saved')

# Параметры сигнала
SAMPLING_RATE = 700

# Параметры скользящего окна
WINDOW_SIZE_SEC = 60
STEP_SIZE_SEC = 30
MIN_RR_PER_WINDOW = 30  # Минимальное кол-во ударов в окне, чтобы не считать его шумом

DEFAULT_SUBJECTS =['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

for d in[DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)