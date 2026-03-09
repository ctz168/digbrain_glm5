"""
DigBrain Core Module
核心处理模块 - 实现类脑计算的核心功能
"""

from .brain import DigBrain, BrainConfig
from .stream import StreamProcessor, StreamConfig
from .attention import BrainAttention, AttentionConfig
from .neuron import NeuronLayer, NeuronConfig
from .streaming_reasoner import StreamingReasoner, StreamingConfig, ReasoningStep

__all__ = [
    'DigBrain', 'BrainConfig',
    'StreamProcessor', 'StreamConfig',
    'BrainAttention', 'AttentionConfig',
    'NeuronLayer', 'NeuronConfig',
    'StreamingReasoner', 'StreamingConfig', 'ReasoningStep'
]
