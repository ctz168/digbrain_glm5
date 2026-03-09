"""
DigBrain Configuration
系统配置
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    path: Optional[str] = None
    device: str = "auto"
    precision: str = "fp16"
    max_length: int = 4096


@dataclass
class MemoryConfig:
    """记忆配置"""
    enabled: bool = True
    capacity: int = 10000
    storage_backend: str = "sqlite"
    storage_path: str = "./data/memory"
    embedding_dim: int = 768


@dataclass
class STDPConfig:
    """STDP配置"""
    enabled: bool = True
    learning_rate: float = 0.01
    time_window: float = 20.0


@dataclass
class StreamConfig:
    """流处理配置"""
    refresh_rate: float = 30.0
    chunk_size: int = 64
    max_context_length: int = 4096


@dataclass
class APIConfig:
    """API配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class WebConfig:
    """Web配置"""
    host: str = "0.0.0.0"
    port: int = 3000


@dataclass
class DigBrainConfig:
    """DigBrain总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    stdp: STDPConfig = field(default_factory=STDPConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    api: APIConfig = field(default_factory=APIConfig)
    web: WebConfig = field(default_factory=WebConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'DigBrainConfig':
        """从YAML文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DigBrainConfig':
        """从字典创建配置"""
        return cls(
            model=ModelConfig(**data.get('model', {})),
            memory=MemoryConfig(**data.get('memory', {})),
            stdp=STDPConfig(**data.get('stdp', {})),
            stream=StreamConfig(**data.get('stream', {})),
            api=APIConfig(**data.get('api', {})),
            web=WebConfig(**data.get('web', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)
    
    def save(self, path: str) -> None:
        """保存配置"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# 默认配置实例
default_config = DigBrainConfig()
