#!/usr/bin/env python3
"""
初始化记忆存储
"""

import asyncio
import os
from pathlib import Path


async def init_memory(storage_path: str = "./data/memory"):
    """初始化记忆存储"""
    from digbrain.memory import HippocampusMemory, MemoryConfig
    
    # 创建目录
    Path(storage_path).mkdir(parents=True, exist_ok=True)
    
    # 初始化记忆系统
    config = MemoryConfig(storage_path=storage_path)
    memory = HippocampusMemory(config)
    await memory.initialize()
    
    # 添加示例记忆
    sample_memories = [
        {
            "content": "DigBrain是一个类脑智能系统，模拟人脑的信息处理机制。",
            "memory_type": "semantic",
            "source": "system"
        },
        {
            "content": "系统支持高刷新率流式处理，模拟人脑毫秒级处理速度。",
            "memory_type": "semantic",
            "source": "system"
        },
        {
            "content": "记忆系统模拟海马体功能，支持短期记忆和长期记忆的转换。",
            "memory_type": "semantic",
            "source": "system"
        }
    ]
    
    for mem in sample_memories:
        await memory.store(mem)
    
    print(f"Memory system initialized at {storage_path}")
    print(f"Stored {len(sample_memories)} sample memories")
    
    await memory.close()


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "./data/memory"
    asyncio.run(init_memory(path))
