#!/usr/bin/env python3
"""
记忆系统优化脚本
优化海马体记忆存储和检索参数
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


async def optimize_memory_system():
    """优化记忆系统"""
    print("\n" + "="*60)
    print("  Memory System Optimization")
    print("="*60)
    
    from memory.hippocampus import HippocampusMemory, MemoryConfig
    
    # 测试不同参数配置
    configs = [
        {'name': 'default', 'capacity': 10000, 'short_term_duration': 30.0, 'consolidation_threshold': 3},
        {'name': 'high_capacity', 'capacity': 50000, 'short_term_duration': 30.0, 'consolidation_threshold': 3},
        {'name': 'fast_consolidation', 'capacity': 10000, 'short_term_duration': 10.0, 'consolidation_threshold': 2},
        {'name': 'slow_forgetting', 'capacity': 10000, 'short_term_duration': 60.0, 'consolidation_threshold': 5},
    ]
    
    # 测试数据
    test_memories = [
        {"content": "人工智能是计算机科学的一个分支", "memory_type": "semantic"},
        {"content": "机器学习是AI的核心技术", "memory_type": "semantic"},
        {"content": "深度学习使用多层神经网络", "memory_type": "semantic"},
        {"content": "自然语言处理让机器理解人类语言", "memory_type": "semantic"},
        {"content": "计算机视觉让机器看懂图像", "memory_type": "semantic"},
        {"content": "强化学习通过奖励训练智能体", "memory_type": "semantic"},
        {"content": "神经网络模拟人脑神经元连接", "memory_type": "semantic"},
        {"content": "Transformer架构改变了NLP领域", "memory_type": "semantic"},
        {"content": "GPT是生成式预训练模型", "memory_type": "semantic"},
        {"content": "BERT是双向编码器表示", "memory_type": "semantic"},
    ]
    
    test_queries = [
        "什么是人工智能？",
        "机器学习是什么？",
        "深度学习的原理",
        "自然语言处理",
        "神经网络",
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting config: {config['name']}")
        print("-"*40)
        
        # 创建记忆系统
        mem_config = MemoryConfig(
            capacity=config['capacity'],
            short_term_duration=config['short_term_duration'],
            consolidation_threshold=config['consolidation_threshold'],
            storage_path=f"./data/memory_test/{config['name']}"
        )
        
        memory = HippocampusMemory(mem_config)
        await memory.initialize()
        
        # 存储测试记忆
        start_store = time.time()
        for mem in test_memories:
            await memory.store(mem)
        store_time = time.time() - start_store
        
        # 检索测试
        start_retrieve = time.time()
        total_relevance = 0
        for query in test_queries:
            results_list = await memory.retrieve(query, top_k=3)
            if results_list:
                total_relevance += sum(r.get('similarity', 0) for r in results_list)
        retrieve_time = time.time() - start_retrieve
        
        # 获取统计
        stats = memory.get_stats()
        
        result = {
            'config': config['name'],
            'store_time': store_time,
            'retrieve_time': retrieve_time,
            'avg_relevance': total_relevance / len(test_queries),
            'total_memories': stats['total_memories'],
            'neural_growth': stats.get('neural_growth', {})
        }
        
        results.append(result)
        
        print(f"  Store time: {store_time:.3f}s")
        print(f"  Retrieve time: {retrieve_time:.3f}s")
        print(f"  Avg relevance: {result['avg_relevance']:.3f}")
        print(f"  Total memories: {stats['total_memories']}")
        
        await memory.close()
    
    # 找出最佳配置
    best_config = max(results, key=lambda x: x['avg_relevance'])
    
    print("\n" + "="*60)
    print("  Optimization Results")
    print("="*60)
    
    for r in results:
        marker = "★" if r['config'] == best_config['config'] else " "
        print(f"{marker} {r['config']:20s}: relevance={r['avg_relevance']:.3f}, "
              f"store={r['store_time']:.3f}s, retrieve={r['retrieve_time']:.3f}s")
    
    print(f"\nBest configuration: {best_config['config']}")
    
    return {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'best_config': best_config['config']
    }


async def train_memory_associations():
    """训练记忆关联"""
    print("\n" + "="*60)
    print("  Memory Association Training")
    print("="*60)
    
    from memory.hippocampus import HippocampusMemory, MemoryConfig
    
    config = MemoryConfig(storage_path="./data/memory_optimized")
    memory = HippocampusMemory(config)
    await memory.initialize()
    
    # 存储相关记忆
    related_memories = [
        {"content": "Python是一种编程语言", "memory_type": "semantic"},
        {"content": "Python支持面向对象编程", "memory_type": "semantic"},
        {"content": "Python有丰富的库生态系统", "memory_type": "semantic"},
        {"content": "NumPy是Python的科学计算库", "memory_type": "semantic"},
        {"content": "Pandas是Python的数据分析库", "memory_type": "semantic"},
        {"content": "TensorFlow是深度学习框架", "memory_type": "semantic"},
        {"content": "PyTorch是另一个深度学习框架", "memory_type": "semantic"},
        {"content": "机器学习需要大量数据", "memory_type": "semantic"},
        {"content": "深度学习需要GPU加速", "memory_type": "semantic"},
        {"content": "神经网络需要反向传播训练", "memory_type": "semantic"},
    ]
    
    memory_ids = []
    for mem in related_memories:
        mem_id = await memory.store(mem)
        memory_ids.append(mem_id)
    
    # 建立关联
    associations = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # Python相关
        (3, 4),  # NumPy和Pandas
        (5, 6),  # TensorFlow和PyTorch
        (7, 8), (7, 9), (8, 9),  # ML/DL相关
    ]
    
    for i, j in associations:
        await memory.associate(memory_ids[i], memory_ids[j])
    
    # 测试联想检索
    print("\nTesting associative retrieval...")
    
    test_cases = [
        ("Python编程", memory_ids[0]),
        ("深度学习框架", memory_ids[5]),
        ("数据分析", memory_ids[4]),
    ]
    
    for query, start_id in test_cases:
        print(f"\nQuery: {query}")
        associated = await memory.get_associated_memories(start_id, depth=2)
        print(f"  Found {len(associated)} associated memories:")
        for mem in associated[:3]:
            print(f"    - {mem['content'][:50]}...")
    
    stats = memory.get_stats()
    await memory.close()
    
    return stats


async def main():
    """主函数"""
    # 优化记忆系统参数
    optimization_results = await optimize_memory_system()
    
    # 训练记忆关联
    association_stats = await train_memory_associations()
    
    # 保存结果
    results = {
        'optimization': optimization_results,
        'associations': association_stats,
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = './results/training/memory_optimization.json'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == '__main__':
    asyncio.run(main())
