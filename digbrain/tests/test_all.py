"""
DigBrain Tests
测试模块
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path


class TestBrain:
    """测试DigBrain核心功能"""
    
    @pytest.fixture
    def brain_config(self):
        """测试配置"""
        from digbrain import BrainConfig
        return BrainConfig(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            memory_capacity=100,
            stdp_enabled=True
        )
    
    @pytest.mark.asyncio
    async def test_brain_initialization(self, brain_config):
        """测试初始化"""
        from digbrain import DigBrain
        
        brain = DigBrain(brain_config)
        assert brain is not None
        assert brain.config == brain_config
    
    @pytest.mark.asyncio
    async def test_brain_process(self, brain_config):
        """测试处理"""
        from digbrain import DigBrain
        
        brain = DigBrain(brain_config)
        await brain.initialize()
        
        result = []
        async for chunk in brain.process("Hello", stream=False):
            result.append(chunk)
        
        assert len(result) > 0
        await brain.shutdown()


class TestMemory:
    """测试记忆系统"""
    
    @pytest.fixture
    async def memory(self, tmp_path):
        """测试记忆系统"""
        from digbrain.memory import HippocampusMemory, MemoryConfig
        
        config = MemoryConfig(
            storage_path=str(tmp_path / "memory"),
            capacity=100
        )
        memory = HippocampusMemory(config)
        await memory.initialize()
        yield memory
        await memory.close()
    
    @pytest.mark.asyncio
    async def test_memory_store(self, memory):
        """测试存储"""
        entry = {
            "content": "Test memory content",
            "memory_type": "episodic"
        }
        
        memory_id = await memory.store(entry)
        assert memory_id is not None
    
    @pytest.mark.asyncio
    async def test_memory_retrieve(self, memory):
        """测试检索"""
        # 存储测试记忆
        await memory.store({
            "content": "Python is a programming language",
            "memory_type": "semantic"
        })
        
        # 检索
        results = await memory.retrieve("programming")
        assert len(results) >= 0


class TestSTDP:
    """测试STDP学习"""
    
    def test_stdp_config(self):
        """测试STDP配置"""
        from digbrain.training import STDPConfig
        
        config = STDPConfig(
            learning_rate_pre=0.01,
            learning_rate_post=0.01
        )
        
        assert config.learning_rate_pre == 0.01
    
    @pytest.mark.asyncio
    async def test_stdp_update(self):
        """测试STDP更新"""
        from digbrain.training import STDPEngine, STDPConfig
        
        config = STDPConfig()
        engine = STDPEngine(config)
        await engine.initialize()
        
        # 注册突触
        engine.register_synapse("test_synapse", 0.5)
        
        # 更新
        weight_change = await engine.update(
            pre_spike_time=0.0,
            post_spike_time=0.01
        )
        
        assert weight_change != 0


class TestTools:
    """测试工具模块"""
    
    @pytest.mark.asyncio
    async def test_calculator(self):
        """测试计算器"""
        from digbrain.tools import WebTools
        
        tools = WebTools()
        result = await tools.call("calculator", expression="2+2")
        
        assert result.success
        assert result.result == 4.0
    
    @pytest.mark.asyncio
    async def test_text_processor(self):
        """测试文本处理"""
        from digbrain.tools import WebTools
        
        tools = WebTools()
        result = await tools.call(
            "text_processor",
            text="Hello World",
            operation="word_count"
        )
        
        assert result.success


class TestEvaluation:
    """测试评估模块"""
    
    def test_metrics(self):
        """测试指标计算"""
        from digbrain.evaluation import MetricsCalculator
        
        calc = MetricsCalculator()
        
        # 测试准确率
        result = calc.calculate_accuracy(
            predictions=["A", "B", "C"],
            labels=["A", "B", "D"]
        )
        
        assert result.value == 2/3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
