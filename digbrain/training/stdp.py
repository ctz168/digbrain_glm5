"""
STDP学习引擎
实现脉冲时序依赖可塑性在线学习
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class STDPConfig:
    """STDP配置"""
    # 学习率
    learning_rate_pre: float = 0.01  # 突触前学习率 (LTP)
    learning_rate_post: float = 0.01  # 突触后学习率 (LTD)
    
    # 时间窗口
    time_window: float = 20.0  # STDP时间窗口 (毫秒)
    tau_plus: float = 20.0  # LTP时间常数
    tau_minus: float = 20.0  # LTD时间常数
    
    # 权重限制
    weight_min: float = 0.0
    weight_max: float = 1.0
    
    # 软边界
    soft_bound: bool = True
    
    # 赫布学习
    hebbian_ratio: float = 1.0  # 赫布学习比例
    
    # 元学习
    meta_learning: bool = True
    meta_lr: float = 0.001


@dataclass
class SynapticTrace:
    """突触追踪"""
    pre_trace: float = 0.0
    post_trace: float = 0.0
    last_pre_spike: float = 0.0
    last_post_spike: float = 0.0


class STDPEngine:
    """
    STDP学习引擎
    
    实现脉冲时序依赖可塑性：
    - LTP: 突触前先于突触后 → 权重增强
    - LTD: 突触后先于突触前 → 权重抑制
    
    支持在线学习和元学习
    """
    
    def __init__(self, config: Optional[STDPConfig] = None):
        self.config = config or STDPConfig()
        
        # 突触追踪
        self._traces: Dict[str, SynapticTrace] = defaultdict(SynapticTrace)
        
        # 权重存储
        self._weights: Dict[str, np.ndarray] = {}
        
        # 学习历史
        self._weight_history: List[Dict] = []
        
        # 元学习参数
        self._meta_params: Dict[str, float] = {
            "lr_pre": self.config.learning_rate_pre,
            "lr_post": self.config.learning_rate_post
        }
        
        # 统计
        self._stats = {
            "total_updates": 0,
            "ltp_events": 0,
            "ltd_events": 0,
            "avg_weight_change": 0.0
        }
    
    async def initialize(self) -> None:
        """初始化STDP引擎"""
        logger.info(f"STDPEngine initialized with window={self.config.time_window}ms")
    
    def register_synapse(
        self,
        synapse_id: str,
        initial_weight: float = 0.5
    ) -> None:
        """
        注册突触
        
        Args:
            synapse_id: 突触ID
            initial_weight: 初始权重
        """
        if synapse_id not in self._weights:
            self._weights[synapse_id] = np.array([initial_weight])
            self._traces[synapse_id] = SynapticTrace()
    
    def register_layer(
        self,
        layer_id: str,
        weight_matrix: np.ndarray
    ) -> None:
        """
        注册权重层
        
        Args:
            layer_id: 层ID
            weight_matrix: 权重矩阵
        """
        self._weights[layer_id] = weight_matrix.copy()
    
    async def update(
        self,
        pre_spike_time: float,
        post_spike_time: float,
        context: Optional[Any] = None,
        synapse_id: Optional[str] = None
    ) -> float:
        """
        更新STDP权重
        
        Args:
            pre_spike_time: 突触前脉冲时间
            post_spike_time: 突触后脉冲时间
            context: 上下文信息
            synapse_id: 可选的特定突触ID
            
        Returns:
            权重变化量
        """
        self._stats["total_updates"] += 1
        
        # 计算时间差
        delta_t = post_spike_time - pre_spike_time
        delta_t_ms = delta_t * 1000  # 转换为毫秒
        
        # 计算权重变化
        weight_change = self._compute_weight_change(delta_t_ms)
        
        # 应用权重变化
        if synapse_id and synapse_id in self._weights:
            self._apply_weight_change(synapse_id, weight_change)
        else:
            # 应用到所有权重
            for sid in self._weights:
                self._apply_weight_change(sid, weight_change)
        
        # 更新追踪
        self._update_traces(pre_spike_time, post_spike_time, synapse_id)
        
        # 元学习更新
        if self.config.meta_learning:
            self._meta_update(weight_change, context)
        
        # 记录历史
        self._weight_history.append({
            "timestamp": time.time(),
            "delta_t": delta_t_ms,
            "weight_change": float(weight_change),
            "synapse_id": synapse_id
        })
        
        return weight_change
    
    def _compute_weight_change(self, delta_t: float) -> float:
        """
        计算权重变化
        
        STDP学习窗口：
        - Δt > 0: LTP (pre在post之前)
        - Δt < 0: LTD (post在pre之前)
        """
        if delta_t > 0:
            # LTP: 突触前先发放
            self._stats["ltp_events"] += 1
            change = self._meta_params["lr_pre"] * np.exp(-delta_t / self.config.tau_plus)
        else:
            # LTD: 突触后先发放
            self._stats["ltd_events"] += 1
            change = -self._meta_params["lr_post"] * np.exp(delta_t / self.config.tau_minus)
        
        # 赫布学习调制
        change *= self.config.hebbian_ratio
        
        return change
    
    def _apply_weight_change(self, synapse_id: str, change: float) -> None:
        """应用权重变化"""
        weights = self._weights[synapse_id]
        
        if self.config.soft_bound:
            # 软边界：权重接近边界时学习率降低
            if change > 0:
                # LTP
                effective_lr = (self.config.weight_max - weights)
                weights += change * effective_lr
            else:
                # LTD
                effective_lr = (weights - self.config.weight_min)
                weights += change * effective_lr
        else:
            # 硬边界
            weights += change
            weights = np.clip(
                weights,
                self.config.weight_min,
                self.config.weight_max
            )
        
        self._weights[synapse_id] = weights
        
        # 更新平均权重变化统计
        self._stats["avg_weight_change"] = (
            self._stats["avg_weight_change"] * (self._stats["total_updates"] - 1) +
            abs(change)
        ) / self._stats["total_updates"]
    
    def _update_traces(
        self,
        pre_time: float,
        post_time: float,
        synapse_id: Optional[str] = None
    ) -> None:
        """更新突触追踪"""
        if synapse_id:
            trace = self._traces[synapse_id]
            trace.pre_trace = pre_time
            trace.post_trace = post_time
            trace.last_pre_spike = pre_time
            trace.last_post_spike = post_time
    
    def _meta_update(self, weight_change: float, context: Optional[Any]) -> None:
        """
        元学习更新
        
        根据学习效果调整学习率
        """
        # 简化的元学习：根据权重变化调整学习率
        if abs(weight_change) > 0.1:
            # 学习效果明显，增加学习率
            self._meta_params["lr_pre"] = min(
                0.1,
                self._meta_params["lr_pre"] * (1 + self.config.meta_lr)
            )
            self._meta_params["lr_post"] = min(
                0.1,
                self._meta_params["lr_post"] * (1 + self.config.meta_lr)
            )
        elif abs(weight_change) < 0.001:
            # 学习效果不明显，降低学习率
            self._meta_params["lr_pre"] = max(
                0.001,
                self._meta_params["lr_pre"] * (1 - self.config.meta_lr)
            )
            self._meta_params["lr_post"] = max(
                0.001,
                self._meta_params["lr_post"] * (1 - self.config.meta_lr)
            )
    
    def get_weights(self, synapse_id: str) -> Optional[np.ndarray]:
        """获取突触权重"""
        return self._weights.get(synapse_id)
    
    def get_all_weights(self) -> Dict[str, np.ndarray]:
        """获取所有权重"""
        return dict(self._weights)
    
    def set_weights(self, synapse_id: str, weights: np.ndarray) -> None:
        """设置权重"""
        self._weights[synapse_id] = weights.copy()
    
    async def save_state(self, path: Path) -> None:
        """保存状态"""
        state = {
            "config": {
                "learning_rate_pre": self.config.learning_rate_pre,
                "learning_rate_post": self.config.learning_rate_post,
                "time_window": self.config.time_window,
                "tau_plus": self.config.tau_plus,
                "tau_minus": self.config.tau_minus
            },
            "meta_params": self._meta_params,
            "stats": self._stats
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # 保存权重
        weights_path = path.parent / "weights.npz"
        np.savez(weights_path, **self._weights)
        
        logger.info(f"STDP state saved to {path}")
    
    async def load_state(self, path: Path) -> None:
        """加载状态"""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self._meta_params = state.get("meta_params", self._meta_params)
        self._stats = state.get("stats", self._stats)
        
        # 加载权重
        weights_path = path.parent / "weights.npz"
        if weights_path.exists():
            data = np.load(weights_path)
            for key in data.files:
                self._weights[key] = data[key]
        
        logger.info(f"STDP state loaded from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "meta_params": self._meta_params,
            "num_synapses": len(self._weights)
        }
    
    def reset(self) -> None:
        """重置状态"""
        self._traces.clear()
        self._weight_history.clear()
        self._stats = {
            "total_updates": 0,
            "ltp_events": 0,
            "ltd_events": 0,
            "avg_weight_change": 0.0
        }


class TripletSTDP(STDPEngine):
    """
    三脉冲STDP
    
    更精确的STDP模型，考虑三个脉冲的时序关系
    """
    
    def __init__(self, config: Optional[STDPConfig] = None):
        super().__init__(config)
        
        # 三脉冲参数
        self.triplet_tau = 100.0  # 三脉冲时间常数
        self.triplet_factor = 1.5  # 三脉冲增强因子
    
    def _compute_weight_change(self, delta_t: float) -> float:
        """计算三脉冲权重变化"""
        # 基础STDP
        base_change = super()._compute_weight_change(delta_t)
        
        # 三脉冲增强
        if abs(delta_t) < self.triplet_tau:
            triplet_boost = self.triplet_factor * np.exp(-abs(delta_t) / self.triplet_tau)
            base_change *= (1 + triplet_boost)
        
        return base_change


class RewardModulatedSTDP(STDPEngine):
    """
    奖励调制STDP
    
    结合奖励信号进行学习
    """
    
    def __init__(self, config: Optional[STDPConfig] = None):
        super().__init__(config)
        
        # 奖励追踪
        self._reward_trace: float = 0.0
        self._reward_tau: float = 1000.0  # 奖励时间常数
    
    def set_reward(self, reward: float) -> None:
        """设置奖励信号"""
        self._reward_trace = reward
    
    def _compute_weight_change(self, delta_t: float) -> float:
        """计算奖励调制权重变化"""
        # 基础STDP
        base_change = super()._compute_weight_change(delta_t)
        
        # 奖励调制
        modulated_change = base_change * (1 + self._reward_trace)
        
        return modulated_change
    
    def update_reward_trace(self, reward: float, dt: float = 1.0) -> None:
        """更新奖励追踪"""
        decay = np.exp(-dt / self._reward_tau)
        self._reward_trace = self._reward_trace * decay + reward * (1 - decay)
