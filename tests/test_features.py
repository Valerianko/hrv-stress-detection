import numpy as np
import pytest
from src.features import extract_features


def test_extract_features_returns_correct_keys():
    # Тестируем, что функция возвращает словарь со всеми нужными ключами
    # Создаем фиктивный массив RR-интервалов (например, пульс ~75 уд/мин)
    dummy_rr = np.array([800, 810, 795, 805, 800, 790] * 10)  # 60 ударов

    features = extract_features(dummy_rr)

    # Проверки (assert)
    assert isinstance(features, dict), "Должен возвращаться словарь"
    assert 'mean_rr' in features, "Отсутствует ключ mean_rr"
    assert 'sdnn' in features, "Отсутствует ключ sdnn"
    assert 'sampen' in features, "Отсутствует ключ sampen"


def test_extract_features_math_logic():
    dummy_rr = np.array([1000, 1000, 1000, 1000] * 10)  # Идеально ровный ритм
    features = extract_features(dummy_rr)

    assert features['mean_rr'] == 1000.0, "Среднее вычислено неверно"
    assert features['sdnn'] == 0.0, "Для ровного ритма SDNN должен быть 0"
    assert features['rmssd'] == 0.0, "Для ровного ритма RMSSD должен быть 0"