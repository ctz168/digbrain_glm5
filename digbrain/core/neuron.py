"""
神经元层模块
模拟生物神经元的计算特性
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class NeuronType(Enum):
    """神经元类型"""
    EXCITATORY = "excitatory"  # 兴奋性
    INHIBITORY = "inhibitory"  # 抑制性
    MODULATORY = "modulatory"  # 调制性


@dataclass
class NeuronConfig:
    """神经元配置"""
    num_neurons: int = 1000
    input_size: int = 768
    hidden_size: int = 512
    
    # 神经元参数
    resting_potential: float = -70.0  # 静息电位 (mV)
    threshold: float = -55.0  # 阈值电位 (mV)
    reset_potential: float = -75.0  # 重置电位 (mV)
    refractory_period: float = 2.0  # 不应期 (ms)
    
    # STDP参数
    stdp_enabled: bool = True
    stdp_lr_pre: float = 0.01  # 突触前学习率
    stdp_lr_post: float = 0.01  # 突触后学习率
    stdp_tau: float = 20.0  # STDP时间常数


class SpikingNeuron:
    """
    脉冲神经元
    
    实现LIF (Leaky Integrate-and-Fire) 模型：
    - 膜电位积分
    - 阈值触发
    - 不应期
    - STDP学习
    """
    
    def __init__(self, config: Optional[NeuronConfig] = None):
        self.config = config or NeuronConfig()
        
        # 神经元状态
        self.membrane_potential = np.full(
            self.config.num_neurons,
            self.config.resting_potential
        )
        self.last_spike_time = np.zeros(self.config.num_neurons)
        self.refractory_until = np.zeros(self.config.num_neurons)
        
        # 突触权重
        self.weights = np.random.randn(
            self.config.input_size,
            self.config.num_neurons
        ) * 0.1
        
        # STDP追踪
        self.pre_trace = np.zeros(self.config.input_size)
        self.post_trace = np.zeros(self.config.num_neurons)
        
        # 统计
        self.spike_count = np.zeros(self.config.num_neurons)
        self.total_spikes = 0
    
    def forward(
        self,
        input_current: np.ndarray,
        time_ms: float,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        前向传播
        
        Args:
            input_current: 输入电流 [batch, input_size]
            time_ms: 当前时间 (毫秒)
            dt: 时间步长
            
        Returns:
            脉冲输出 [batch, num_neurons]
        """
        # 计算突触电流
        synaptic_current = np.matmul(input_current, self.weights)
        
        # 检查不应期
        in_refractory = time_ms < self.refractory_until
        
        # 膜电位衰减 (LIF)
        tau_m = 20.0  # 膜时间常数
        decay = np.exp(-dt / tau_m)
        self.membrane_potential = (
            self.membrane_potential * decay +
            synaptic_current * (1 - decay)
        )
        
        # 重置不应期内的神经元
        self.membrane_potential[in_refractory] = self.config.reset_potential
        
        # 检测脉冲
        spikes = self.membrane_potential >= self.config.threshold
        
        # 更新脉冲神经元状态
        self.membrane_potential[spikes] = self.config.reset_potential
        self.refractory_until[spikes] = time_ms + self.config.refractory_period
        self.last_spike_time[spikes] = time_ms
        
        # 更新统计
        self.spike_count[spikes] += 1
        self.total_spikes += np.sum(spikes)
        
        # STDP更新
        if self.config.stdp_enabled:
            self._update_stdp(input_current, spikes, time_ms)
        
        return spikes.astype(float)
    
    def _update_stdp(
        self,
        pre_activity: np.ndarray,
        post_spikes: np.ndarray,
        time_ms: float
    ) -> None:
        """
        STDP权重更新
        
        实现：
        - LTP: 突触前先于突触后 → 增强
        - LTD: 突触后先于突触前 → 抑制
        """
        # 更新追踪变量
        tau_stdp = self.config.stdp_tau
        
        # 突触前追踪
        self.pre_trace = (
            self.pre_trace * np.exp(-1 / tau_stdp) +
            pre_activity.mean(axis=0)
        )
        
        # 突触后追踪
        self.post_trace = (
            self.post_trace * np.exp(-1 / tau_stdp) +
            post_spikes
        )
        
        # 计算权重变化
        # LTP: pre → post
        if np.any(post_spikes):
            delta_w_ltp = (
                self.config.stdp_lr_pre *
                np.outer(self.pre_trace, post_spikes)
            )
            self.weights += delta_w_ltp
        
        # LTD: post → pre
        if np.any(pre_activity):
            delta_w_ltd = (
                -self.config.stdp_lr_post *
                np.outer(pre_activity.mean(axis=0), self.post_trace)
            )
            self.weights += delta_w_ltd
        
        # 权重裁剪
        self.weights = np.clip(self.weights, -1.0, 1.0)
    
    def reset(self) -> None:
        """重置神经元状态"""
        self.membrane_potential = np.full(
            self.config.num_neurons,
            self.config.resting_potential
        )
        self.last_spike_time = np.zeros(self.config.num_neurons)
        self.refractory_until = np.zeros(self.config.num_neurons)
        self.pre_trace = np.zeros(self.config.input_size)
        self.post_trace = np.zeros(self.config.num_neurons)


class NeuronLayer:
    """
    神经元层
    
    管理多个神经元群体，支持：
    - 兴奋性/抑制性神经元
    - 层间连接
    - 神经可塑性
    """
    
    def __init__(self, config: Optional[NeuronConfig] = None):
        self.config = config or NeuronConfig()
        
        # 创建神经元群体
        # 80% 兴奋性, 20% 抑制性 (符合生物比例)
        num_excitatory = int(self.config.num_neurons * 0.8)
        num_inhibitory = self.config.num_neurons - num_excitatory
        
        self.excitatory = SpikingNeuron(NeuronConfig(
            num_neurons=num_excitatory,
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size
        ))
        
        self.inhibitory = SpikingNeuron(NeuronConfig(
            num_neurons=num_inhibitory,
            input_size=num_excitatory,
            hidden_size=self.config.hidden_size // 2
        ))
        
        # 层间连接
        self.e_to_i_weights = np.random.randn(num_excitatory, num_inhibitory) * 0.1
        self.i_to_e_weights = np.random.randn(num_inhibitory, num_excitatory) * -0.1
        
        # 活动记录
        self.activity_history: List[np.ndarray] = []
    
    def forward(
        self,
        input_current: np.ndarray,
        time_ms: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播
        
        Args:
            input_current: 输入电流
            time_ms: 当前时间
            
        Returns:
            兴奋性和抑制性神经元的脉冲输出
        """
        # 兴奋性神经元
        e_spikes = self.excitatory.forward(input_current, time_ms)
        
        # 兴奋性 → 抑制性
        i_input = np.matmul(e_spikes, self.e_to_i_weights)
        i_spikes = self.inhibitory.forward(i_input, time_ms)
        
        # 抑制性 → 兴奋性 (反馈抑制)
        inhibition = np.matmul(i_spikes, self.i_to_e_weights)
        
        # 记录活动
        self.activity_history.append(np.concatenate([e_spikes, i_spikes]))
        
        return e_spikes, i_spikes
    
    def get_activity_rate(self, window: int = 100) -> np.ndarray:
        """获取最近的活动率"""
        if len(self.activity_history) == 0:
            return np.zeros(self.config.num_neurons)
        
        recent = self.activity_history[-window:]
        return np.mean(recent, axis=0)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """获取所有权重"""
        return {
            "excitatory": self.excitatory.weights,
            "inhibitory": self.inhibitory.weights,
            "e_to_i": self.e_to_i_weights,
            "i_to_e": self.i_to_e_weights
        }
    
    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """设置权重"""
        if "excitatory" in weights:
            self.excitatory.weights = weights["excitatory"]
        if "inhibitory" in weights:
            self.inhibitory.weights = weights["inhibitory"]
        if "e_to_i" in weights:
            self.e_to_i_weights = weights["e_to_i"]
        if "i_to_e" in weights:
            self.i_to_e_weights = weights["i_to_e"]
    
    def reset(self) -> None:
        """重置层状态"""
        self.excitatory.reset()
        self.inhibitory.reset()
        self.activity_history.clear()


class NeuralPopulation:
    """
    神经元群体
    
    模拟大脑中的神经元群体动力学
    """
    
    def __init__(self, num_neurons: int = 1000):
        self.num_neurons = num_neurons
        
        # 神经元状态
        self.firing_rates = np.zeros(num_neurons)
        self.adaptation = np.zeros(num_neurons)
        
        # 连接矩阵
        self.connectivity = self._init_connectivity()
    
    def _init_connectivity(self) -> np.ndarray:
        """初始化连接矩阵（稀疏）"""
        # 随机稀疏连接
        connectivity = np.random.randn(self.num_neurons, self.num_neurons) * 0.1
        # 稀疏化 (10% 连接率)
        mask = np.random.random((self.num_neurons, self.num_neurons)) > 0.1
        connectivity[mask] = 0
        return connectivity
    
    def update(
        self,
        external_input: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        更新群体状态
        
        Args:
            external_input: 外部输入
            dt: 时间步长
            
        Returns:
            当前发放率
        """
        # 群体输入
        recurrent = np.matmul(self.firing_rates, self.connectivity)
        total_input = external_input + recurrent
        
        # 发放率更新 (简化模型)
        tau = 20.0  # 时间常数
        self.firing_rates = (
            self.firing_rates * np.exp(-dt / tau) +
            self._activation(total_input) * (1 - np.exp(-dt / tau))
        )
        
        # 适应性
        self.adaptation += 0.01 * self.firing_rates
        self.firing_rates *= np.exp(-self.adaptation * dt)
        
        return self.firing_rates
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """激活函数"""
        # ReLU-like
        return np.maximum(0, x)
    
    def get_population_activity(self) -> Dict[str, float]:
        """获取群体活动统计"""
        return {
            "mean_rate": np.mean(self.firing_rates),
            "max_rate": np.max(self.firing_rates),
            "active_fraction": np.mean(self.firing_rates > 0.1),
            "synchrony": self._compute_synchrony()
        }
    
    def _compute_synchrony(self) -> float:
        """计算同步性"""
        if np.std(self.firing_rates) < 1e-6:
            return 0.0
        return np.mean(self.firing_rates) / (np.std(self.firing_rates) + 1e-6)
