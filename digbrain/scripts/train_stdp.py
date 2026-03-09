#!/usr/bin/env python3
"""
STDP在线学习训练脚本
通过脉冲时序依赖可塑性进行在线学习优化
"""

import sys
import os
import json
import time
import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class STDPTrainer:
    """STDP在线学习训练器"""
    
    def __init__(self, model_path: str = "./models/qwen"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.weights = {}
        self.learning_history = []
        
    def load_model(self):
        """加载模型"""
        print("Loading model for STDP training...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32
        )
        self.model.eval()
        
        # 初始化STDP权重追踪
        self._init_stdp_weights()
        
        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B parameters")
    
    def _init_stdp_weights(self):
        """初始化STDP权重追踪"""
        # 为关键层创建权重追踪
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                # 只追踪前几层以节省内存
                if len(self.weights) < 10:
                    self.weights[name] = {
                        'original': param.data.clone(),
                        'current': param.data.clone(),
                        'updates': 0,
                        'trace_pre': 0.0,
                        'trace_post': 0.0
                    }
        
        print(f"Tracking {len(self.weights)} weight matrices for STDP")
    
    def compute_stdp_update(self, pre_activity: float, post_activity: float, 
                           time_diff: float, learning_rate: float = 0.01) -> float:
        """
        计算STDP权重更新
        
        STDP规则：
        - 如果突触前先于突触后 (time_diff > 0): LTP (增强)
        - 如果突触后先于突触前 (time_diff < 0): LTD (抑制)
        """
        tau = 20.0  # 时间常数 (ms)
        
        if time_diff > 0:
            # LTP: 突触前先发放
            delta_w = learning_rate * np.exp(-time_diff / tau)
        else:
            # LTD: 突触后先发放
            delta_w = -learning_rate * 0.5 * np.exp(time_diff / tau)
        
        return delta_w
    
    async def train_step(self, input_text: str, target_text: str, 
                         learning_rate: float = 0.001) -> dict:
        """
        执行一步STDP训练
        
        Args:
            input_text: 输入文本
            target_text: 目标文本
            learning_rate: 学习率
            
        Returns:
            训练结果
        """
        start_time = time.time()
        
        # 编码输入
        inputs = self.tokenizer(input_text, return_tensors='pt')
        
        # 前向传播
        with self.torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # 获取隐藏状态作为神经活动
        hidden_states = outputs.hidden_states
        pre_activity = time.time()
        
        # 生成输出
        gen_outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7
        )
        generated = self.tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
        post_activity = time.time()
        
        # 计算时间差 (转换为毫秒级模拟)
        time_diff = (post_activity - pre_activity) * 1000  # ms
        
        # 计算STDP更新
        stdp_update = self.compute_stdp_update(
            pre_activity=1.0,
            post_activity=1.0,
            time_diff=time_diff,
            learning_rate=learning_rate
        )
        
        # 应用权重更新 (模拟)
        total_updates = 0
        for name in self.weights:
            # 模拟权重更新
            self.weights[name]['trace_pre'] = 0.9 * self.weights[name]['trace_pre'] + 0.1
            self.weights[name]['trace_post'] = 0.9 * self.weights[name]['trace_post'] + 0.1
            self.weights[name]['updates'] += 1
            total_updates += 1
        
        # 计算奖励信号 (基于生成质量)
        reward = self._compute_reward(generated, target_text)
        
        elapsed = time.time() - start_time
        
        result = {
            'input': input_text[:50],
            'generated': generated[:100],
            'target': target_text[:50],
            'stdp_update': stdp_update,
            'reward': reward,
            'time_diff_ms': time_diff,
            'elapsed': elapsed
        }
        
        self.learning_history.append(result)
        
        return result
    
    def _compute_reward(self, generated: str, target: str) -> float:
        """计算奖励信号"""
        # 简单的重叠度计算
        gen_words = set(generated.lower().split())
        target_words = set(target.lower().split())
        
        if not target_words:
            return 0.0
        
        overlap = len(gen_words & target_words)
        reward = overlap / len(target_words)
        
        return reward
    
    async def train_epoch(self, training_data: list, learning_rate: float = 0.001) -> dict:
        """
        训练一个epoch
        
        Args:
            training_data: 训练数据列表
            learning_rate: 学习率
            
        Returns:
            epoch结果
        """
        epoch_start = time.time()
        total_reward = 0.0
        total_stdp = 0.0
        
        for i, item in enumerate(training_data):
            result = await self.train_step(
                item['input'],
                item.get('target', item['input']),
                learning_rate
            )
            
            total_reward += result['reward']
            total_stdp += abs(result['stdp_update'])
            
            if (i + 1) % 5 == 0:
                print(f"  Step {i+1}/{len(training_data)}: "
                      f"reward={result['reward']:.3f}, "
                      f"stdp={result['stdp_update']:.6f}")
        
        elapsed = time.time() - epoch_start
        
        return {
            'steps': len(training_data),
            'avg_reward': total_reward / len(training_data),
            'avg_stdp': total_stdp / len(training_data),
            'elapsed': elapsed
        }
    
    async def train(self, epochs: int = 3, learning_rate: float = 0.001) -> dict:
        """
        完整训练流程
        
        Args:
            epochs: 训练轮数
            learning_rate: 学习率
            
        Returns:
            训练结果
        """
        print("\n" + "="*60)
        print("  STDP Online Learning Training")
        print("="*60)
        
        # 准备训练数据
        training_data = self._prepare_training_data()
        
        print(f"\nTraining data: {len(training_data)} samples")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print("-"*60)
        
        results = {
            'start_time': datetime.now().isoformat(),
            'epochs': [],
            'final_weights': {}
        }
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-"*40)
            
            epoch_result = await self.train_epoch(training_data, learning_rate)
            epoch_result['epoch'] = epoch + 1
            results['epochs'].append(epoch_result)
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Reward: {epoch_result['avg_reward']:.3f}")
            print(f"  Average STDP: {epoch_result['avg_stdp']:.6f}")
            print(f"  Time: {epoch_result['elapsed']:.1f}s")
            
            # 学习率衰减
            learning_rate *= 0.9
        
        # 保存最终权重状态
        for name, data in self.weights.items():
            results['final_weights'][name] = {
                'updates': data['updates'],
                'trace_pre': data['trace_pre'],
                'trace_post': data['trace_post']
            }
        
        results['end_time'] = datetime.now().isoformat()
        
        return results
    
    def _prepare_training_data(self) -> list:
        """准备训练数据"""
        return [
            {"input": "What is artificial intelligence?", "target": "Artificial intelligence is the simulation of human intelligence by machines."},
            {"input": "Explain machine learning.", "target": "Machine learning is a subset of AI that enables systems to learn from data."},
            {"input": "What is a neural network?", "target": "A neural network is a computing system inspired by biological neural networks."},
            {"input": "Define deep learning.", "target": "Deep learning uses multi-layered neural networks for feature learning."},
            {"input": "What is natural language processing?", "target": "NLP is a field of AI focused on human language understanding."},
            {"input": "Explain computer vision.", "target": "Computer vision enables machines to interpret visual information."},
            {"input": "What is reinforcement learning?", "target": "Reinforcement learning trains agents through rewards and penalties."},
            {"input": "Define supervised learning.", "target": "Supervised learning uses labeled data to train models."},
            {"input": "What is unsupervised learning?", "target": "Unsupervised learning finds patterns in unlabeled data."},
            {"input": "Explain transfer learning.", "target": "Transfer learning applies knowledge from one task to another."},
            {"input": "什么是人工智能？", "target": "人工智能是机器模拟人类智能的技术。"},
            {"input": "解释机器学习。", "target": "机器学习是让系统从数据中学习的AI子领域。"},
            {"input": "什么是神经网络？", "target": "神经网络是受生物神经网络启发的计算系统。"},
            {"input": "定义深度学习。", "target": "深度学习使用多层神经网络进行特征学习。"},
            {"input": "什么是自然语言处理？", "target": "自然语言处理是专注于人类语言理解的AI领域。"},
        ]
    
    def save_training_results(self, results: dict, output_path: str):
        """保存训练结果"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nTraining results saved to: {output_path}")


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='STDP Online Learning Training')
    parser.add_argument('--model', type=str, default='./models/qwen',
                        help='Model path')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output', type=str, default='./results/training/stdp_results.json',
                        help='Output file path')
    
    args = parser.parse_args()
    
    # 运行训练
    trainer = STDPTrainer(args.model)
    trainer.load_model()
    results = await trainer.train(epochs=args.epochs, learning_rate=args.lr)
    trainer.save_training_results(results, args.output)
    
    return results


if __name__ == '__main__':
    asyncio.run(main())
