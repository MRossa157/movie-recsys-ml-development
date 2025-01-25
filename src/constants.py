from enum import Enum


class ItemsFeatureTopKConfig(int, Enum):
    """Конфигурация для ограничения количества топовых значений фич."""
    DIRECTORS_TOP_K = 30
    STUDIOS_TOP_K = 15
