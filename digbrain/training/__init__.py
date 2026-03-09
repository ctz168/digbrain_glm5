"""
DigBrain Training Module
训练模块 - 实现在线STDP学习和离线训练
"""

from .stdp import STDPEngine, STDPConfig
from .online import OnlineLearner, OnlineConfig
from .offline import OfflineTrainer, OfflineConfig

__all__ = [
    'STDPEngine', 'STDPConfig',
    'OnlineLearner', 'OnlineConfig',
    'OfflineTrainer', 'OfflineConfig'
]
