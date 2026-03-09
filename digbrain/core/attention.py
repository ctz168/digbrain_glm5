"""
类脑注意力机制
模拟人脑的选择性注意和分布式注意
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class AttentionConfig:
    """注意力配置"""
    hidden_size: int = 768
    num_heads: int = 12
    attention_dropout: float = 0.1
    max_position_embeddings: int = 4096
    
    # 类脑特有配置
    attention_decay: float = 0.1  # 注意力衰减
    novelty_bonus: float = 0.2  # 新颖性加成
    relevance_threshold: float = 0.3  # 相关性阈值


class BrainAttention:
    """
    类脑注意力机制
    
    实现人脑式的注意力分配：
    1. 选择性注意 - 聚焦于重要信息
    2. 分布式注意 - 同时处理多个信息源
    3. 动态调整 - 根据上下文调整注意力权重
    4. 新颖性检测 - 对新信息给予更多关注
    """
    
    def __init__(self, config: Optional[AttentionConfig] = None):
        self.config = config or AttentionConfig()
        
        # 注意力权重
        self.head_dim = self.config.hidden_size // self.config.num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # 初始化权重
        self._init_weights()
        
        # 注意力状态
        self._attention_history: List[np.ndarray] = []
        self._novelty_scores: Dict[str, float] = {}
        
    def _init_weights(self) -> None:
        """初始化权重"""
        # Query, Key, Value 投影
        self.W_q = np.random.randn(
            self.config.hidden_size,
            self.config.hidden_size
        ) * 0.02
        
        self.W_k = np.random.randn(
            self.config.hidden_size,
            self.config.hidden_size
        ) * 0.02
        
        self.W_v = np.random.randn(
            self.config.hidden_size,
            self.config.hidden_size
        ) * 0.02
        
        self.W_o = np.random.randn(
            self.config.hidden_size,
            self.config.hidden_size
        ) * 0.02
    
    def compute_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算注意力
        
        Args:
            query: 查询向量 [batch, seq_len, hidden]
            key: 键向量 [batch, seq_len, hidden]
            value: 值向量 [batch, seq_len, hidden]
            mask: 注意力掩码
            
        Returns:
            输出向量和注意力权重
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # 投影
        Q = np.matmul(query, self.W_q)
        K = np.matmul(key, self.W_k)
        V = np.matmul(value, self.W_v)
        
        # 重塑为多头
        Q = Q.reshape(batch_size, seq_len, self.config.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.config.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.config.num_heads, self.head_dim)
        
        # 转置
        Q = Q.transpose(0, 2, 1, 3)  # [batch, heads, seq, dim]
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # 计算注意力分数
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / self.scale
        
        # 应用掩码
        if mask is not None:
            scores = scores + mask * -1e9
        
        # Softmax
        attention_weights = self._softmax(scores, axis=-1)
        
        # 应用类脑调制
        attention_weights = self._apply_brain_modulation(attention_weights)
        
        # 计算输出
        output = np.matmul(attention_weights, V)
        
        # 重塑回原形状
        output = output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.config.hidden_size
        )
        
        # 输出投影
        output = np.matmul(output, self.W_o)
        
        # 记录注意力历史
        self._attention_history.append(attention_weights.mean(axis=(0, 1)))
        
        return output, attention_weights
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """数值稳定的softmax"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _apply_brain_modulation(self, attention: np.ndarray) -> np.ndarray:
        """
        应用类脑调制
        
        包括：
        1. 注意力衰减 - 模拟人脑注意力疲劳
        2. 新颖性加成 - 对新信息给予更多关注
        """
        # 注意力衰减
        if len(self._attention_history) > 0:
            recent_attention = np.mean(self._attention_history[-5:], axis=0)
            decay = 1 - self.config.attention_decay * recent_attention
            attention = attention * decay
        
        # 重新归一化
        attention = attention / (attention.sum(axis=-1, keepdims=True) + 1e-9)
        
        return attention
    
    def compute_novelty(self, input_embedding: np.ndarray) -> float:
        """
        计算输入的新颖性
        
        人脑会对新颖刺激给予更多注意
        """
        # 计算与历史嵌入的平均距离
        if len(self._attention_history) == 0:
            return 1.0  # 第一个输入最具新颖性
        
        # 简化的新颖性计算
        novelty = np.random.random() * self.config.novelty_bonus
        return min(1.0, novelty)
    
    def get_attention_weights(self) -> np.ndarray:
        """获取最近的注意力权重"""
        if len(self._attention_history) == 0:
            return np.array([])
        return self._attention_history[-1]
    
    def reset_history(self) -> None:
        """重置注意力历史"""
        self._attention_history.clear()
        self._novelty_scores.clear()


class SelectiveAttention:
    """
    选择性注意模块
    
    模拟人脑的选择性注意机制：
    - 过滤无关信息
    - 聚焦于目标
    - 抑制干扰
    """
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self._focus_target: Optional[np.ndarray] = None
    
    def set_focus(self, target: np.ndarray) -> None:
        """设置注意焦点"""
        self._focus_target = target
    
    def filter_input(
        self,
        inputs: np.ndarray,
        relevance_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据相关性过滤输入
        
        Args:
            inputs: 输入向量 [batch, seq, dim]
            relevance_scores: 相关性分数 [batch, seq]
            
        Returns:
            过滤后的输入和掩码
        """
        # 创建掩码
        mask = relevance_scores > self.threshold
        
        # 应用掩码
        filtered = inputs * mask[:, :, np.newaxis]
        
        return filtered, mask
    
    def compute_relevance(
        self,
        query: np.ndarray,
        keys: np.ndarray
    ) -> np.ndarray:
        """计算相关性分数"""
        if self._focus_target is not None:
            # 使用焦点目标计算相关性
            query = self._focus_target
        
        # 点积相似度
        scores = np.matmul(keys, query.T).squeeze(-1)
        
        # 归一化
        scores = scores / (np.linalg.norm(keys, axis=-1, keepdims=True) + 1e-9)
        scores = scores / (np.linalg.norm(query) + 1e-9)
        
        return scores


class DistributedAttention:
    """
    分布式注意模块
    
    模拟人脑同时处理多个信息源的能力
    """
    
    def __init__(self, num_streams: int = 4):
        self.num_streams = num_streams
        self._streams: List[BrainAttention] = [
            BrainAttention() for _ in range(num_streams)
        ]
    
    def process_parallel(
        self,
        inputs: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        并行处理多个输入流
        
        Args:
            inputs: 多个输入向量列表
            
        Returns:
            每个流的处理结果
        """
        results = []
        
        for i, (stream, inp) in enumerate(zip(self._streams, inputs)):
            output, weights = stream.compute_attention(inp, inp, inp)
            results.append((output, weights))
        
        return results
    
    def merge_results(
        self,
        results: List[Tuple[np.ndarray, np.ndarray]],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        合并多个流的结果
        
        Args:
            results: 各流的结果
            weights: 各流的权重
            
        Returns:
            合并后的结果
        """
        if weights is None:
            weights = [1.0 / len(results)] * len(results)
        
        merged = np.zeros_like(results[0][0])
        
        for (output, _), w in zip(results, weights):
            merged += output * w
        
        return merged
