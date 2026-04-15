import os

# Пути к данным
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models_saved')

# Параметры сигнала
SAMPLING_RATE = 700

# Параметры скользящего окна
WINDOW_SIZE_SEC = 60
STEP_SIZE_SEC = 30
MIN_RR_PER_WINDOW = 30  # Минимальное кол-во ударов в окне, чтобы не считать его шумом

# Субъекты по умолчанию
DEFAULT_SUBJECTS =['S2', 'S3', 'S4', 'S5']