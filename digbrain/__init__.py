"""
DigBrain - 类脑智能系统
主入口模块
"""

__version__ = "1.0.0"
__author__ = "DigBrain Team"

from .core import DigBrain, BrainConfig
from .memory import HippocampusMemory, MemoryConfig
from .training import STDPEngine, STDPConfig
from .tools import ToolManager, ToolConfig

__all__ = [
    'DigBrain', 'BrainConfig',
    'HippocampusMemory', 'MemoryConfig',
    'STDPEngine', 'STDPConfig',
    'ToolManager', 'ToolConfig',
    '__version__'
]


def create_brain(config=None):
    """创建DigBrain实例的便捷函数"""
    return DigBrain(config)
