"""
流式思维链推理模块
实现逐步推理、并行记忆检索、实时工具调用
"""

import asyncio
import time
import json
import re
from typing import Optional, Dict, Any, List, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class ReasoningStep(Enum):
    """推理步骤类型"""
    UNDERSTAND = "understand"      # 理解问题
    ANALYZE = "analyze"           # 分析问题
    RETRIEVE = "retrieve"         # 检索知识
    REASON = "reason"             # 推理
    VERIFY = "verify"             # 验证
    CONCLUDE = "conclude"         # 得出结论


@dataclass
class StreamingConfig:
    """流式推理配置"""
    # 刷新率配置
    refresh_rate: float = 30.0  # Hz
    chunk_tokens: int = 8  # 每次生成的token数
    
    # 思维链配置
    enable_cot: bool = True  # 启用思维链
    max_reasoning_steps: int = 5  # 最大推理步骤
    reasoning_depth: int = 2  # 推理深度
    
    # 并行处理配置
    parallel_memory: bool = True  # 并行记忆检索
    parallel_tools: bool = True  # 并行工具调用
    
    # 记忆检索配置
    memory_top_k: int = 3  # 记忆检索数量
    wiki_search: bool = True  # 维基百科搜索
    
    # 输出配置
    stream_thoughts: bool = True  # 流式输出思考过程
    verbose: bool = True  # 详细输出


@dataclass
class ReasoningState:
    """推理状态"""
    step: ReasoningStep
    content: str
    timestamp: float = field(default_factory=time.time)
    memory_results: List[Dict] = field(default_factory=list)
    tool_results: List[Dict] = field(default_factory=list)
    confidence: float = 0.0


class StreamingReasoner:
    """
    流式思维链推理器
    
    核心特性：
    1. 逐步推理 - 每步生成少量token，模拟人脑思考
    2. 并行检索 - 在推理过程中并行搜索记忆和知识库
    3. 高刷新率 - 30Hz刷新，实时响应
    4. 思维链 - 可视化推理过程
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        memory_system=None,
        tool_manager=None,
        config: Optional[StreamingConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.memory_system = memory_system
        self.tool_manager = tool_manager
        self.config = config or StreamingConfig()
        
        # 推理状态
        self._current_state: Optional[ReasoningState] = None
        self._reasoning_history: List[ReasoningState] = []
        
        # 流式缓冲
        self._output_buffer: deque = deque(maxlen=1000)
        self._thought_buffer: deque = deque(maxlen=100)
        
        # 统计
        self._stats = {
            "total_inferences": 0,
            "total_tokens": 0,
            "total_steps": 0,
            "memory_queries": 0,
            "tool_calls": 0,
            "avg_latency": 0.0
        }
    
    async def stream_reason(
        self,
        query: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式推理主入口
        
        实现逐步推理流程：
        1. 理解问题 → 2. 分析问题 → 3. 检索知识 → 4. 推理 → 5. 验证 → 6. 结论
        
        每一步都是流式输出，同时并行执行记忆检索和工具调用
        """
        self._stats["total_inferences"] += 1
        start_time = time.time()
        
        # 初始化推理状态
        self._current_state = ReasoningState(
            step=ReasoningStep.UNDERSTAND,
            content=""
        )
        
        # 构建初始提示
        system_prompt = self._build_system_prompt()
        full_prompt = f"{system_prompt}\n\n问题: {query}\n\n"
        if context:
            full_prompt = f"背景信息: {context}\n\n{full_prompt}"
        
        # 步骤1: 理解问题
        async for chunk in self._step_understand(query, full_prompt):
            yield chunk
        
        # 步骤2: 分析问题（并行检索记忆）
        memory_task = None
        if self.memory_system and self.config.parallel_memory:
            memory_task = asyncio.create_task(
                self._retrieve_memory(query)
            )
        
        async for chunk in self._step_analyze(query):
            yield chunk
        
        # 等待记忆检索完成
        if memory_task:
            memory_results = await memory_task
            self._current_state.memory_results = memory_results
            self._stats["memory_queries"] += 1
            
            # 输出记忆检索结果
            if memory_results and self.config.stream_thoughts:
                yield {
                    "type": "memory",
                    "content": f"[检索到{len(memory_results)}条相关记忆]",
                    "results": memory_results[:self.config.memory_top_k]
                }
        
        # 步骤3: 检索知识（维基百科）
        wiki_task = None
        if self.config.wiki_search and self.tool_manager:
            wiki_task = asyncio.create_task(
                self._search_wiki(query)
            )
        
        # 步骤4: 推理（核心步骤）
        reasoning_context = self._build_reasoning_context(
            query, 
            self._current_state.memory_results
        )
        
        async for chunk in self._step_reason(query, reasoning_context):
            yield chunk
        
        # 等待维基搜索完成
        if wiki_task:
            wiki_results = await wiki_task
            if wiki_results:
                self._current_state.tool_results = [wiki_results]
                self._stats["tool_calls"] += 1
                
                if self.config.stream_thoughts:
                    yield {
                        "type": "knowledge",
                        "content": f"[知识库: {wiki_results.get('title', 'N/A')}]",
                        "summary": wiki_results.get('summary', '')[:200]
                    }
        
        # 步骤5: 验证
        async for chunk in self._step_verify():
            yield chunk
        
        # 步骤6: 得出结论
        async for chunk in self._step_conclude():
            yield chunk
        
        # 更新统计
        elapsed = time.time() - start_time
        self._stats["avg_latency"] = (
            (self._stats["avg_latency"] * (self._stats["total_inferences"] - 1) + elapsed)
            / self._stats["total_inferences"]
        )
        
        # 保存推理历史
        self._reasoning_history.append(self._current_state)
        
        # 最终输出
        yield {
            "type": "done",
            "stats": {
                "elapsed": elapsed,
                "steps": len(self._reasoning_history),
                "memory_queries": self._stats["memory_queries"],
                "tool_calls": self._stats["tool_calls"]
            }
        }
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return """你是一个智能助手，请逐步思考并回答问题。

思考步骤：
1. 理解：首先理解问题的核心
2. 分析：分解问题的关键要素
3. 检索：回忆相关知识
4. 推理：进行逻辑推理
5. 验证：检查推理是否合理
6. 结论：给出最终答案

请用简洁的语言逐步思考，每一步都要清晰明确。"""
    
    async def _step_understand(
        self,
        query: str,
        prompt: str
    ) -> AsyncGenerator[Dict, None]:
        """步骤1: 理解问题"""
        self._current_state.step = ReasoningStep.UNDERSTAND
        
        understand_prompt = f"{prompt}第一步：理解问题\n思考："
        
        yield {
            "type": "step",
            "step": "understand",
            "content": "[理解问题]\n"
        }
        
        # 流式生成
        async for chunk in self._stream_generate(understand_prompt, max_tokens=50):
            self._current_state.content += chunk
            yield {
                "type": "thought",
                "step": "understand",
                "content": chunk
            }
        
        yield {"type": "step_end", "step": "understand"}
    
    async def _step_analyze(
        self,
        query: str
    ) -> AsyncGenerator[Dict, None]:
        """步骤2: 分析问题"""
        self._current_state.step = ReasoningStep.ANALYZE
        
        analyze_prompt = f"\n第二步：分析问题\n这个问题的关键要素是："
        
        yield {
            "type": "step",
            "step": "analyze",
            "content": "\n[分析问题]\n"
        }
        
        async for chunk in self._stream_generate(analyze_prompt, max_tokens=80):
            yield {
                "type": "thought",
                "step": "analyze",
                "content": chunk
            }
        
        yield {"type": "step_end", "step": "analyze"}
    
    async def _step_reason(
        self,
        query: str,
        context: str
    ) -> AsyncGenerator[Dict, None]:
        """步骤4: 推理（核心）"""
        self._current_state.step = ReasoningStep.REASON
        
        reason_prompt = f"""{context}

第四步：推理
让我一步步推理：

1. 首先，"""
        
        yield {
            "type": "step",
            "step": "reason",
            "content": "\n[推理过程]\n"
        }
        
        # 流式生成推理过程
        async for chunk in self._stream_generate(reason_prompt, max_tokens=200):
            yield {
                "type": "thought",
                "step": "reason",
                "content": chunk
            }
        
        yield {"type": "step_end", "step": "reason"}
    
    async def _step_verify(self) -> AsyncGenerator[Dict, None]:
        """步骤5: 验证"""
        self._current_state.step = ReasoningStep.VERIFY
        
        verify_prompt = "\n第五步：验证\n让我检查推理是否合理："
        
        yield {
            "type": "step",
            "step": "verify",
            "content": "\n[验证]\n"
        }
        
        async for chunk in self._stream_generate(verify_prompt, max_tokens=50):
            yield {
                "type": "thought",
                "step": "verify",
                "content": chunk
            }
        
        yield {"type": "step_end", "step": "verify"}
    
    async def _step_conclude(self) -> AsyncGenerator[Dict, None]:
        """步骤6: 得出结论"""
        self._current_state.step = ReasoningStep.CONCLUDE
        
        conclude_prompt = "\n第六步：结论\n最终答案是："
        
        yield {
            "type": "step",
            "step": "conclude",
            "content": "\n[结论]\n"
        }
        
        async for chunk in self._stream_generate(conclude_prompt, max_tokens=100):
            yield {
                "type": "answer",
                "step": "conclude",
                "content": chunk
            }
        
        yield {"type": "step_end", "step": "conclude"}
    
    async def _stream_generate(
        self,
        prompt: str,
        max_tokens: int = 100
    ) -> AsyncGenerator[str, None]:
        """
        流式生成文本
        
        实现高刷新率的token级流式输出
        """
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        # 使用流式生成
        generated_tokens = 0
        past_key_values = None
        
        with torch.no_grad():
            while generated_tokens < max_tokens:
                # 每次生成少量token
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(self.config.chunk_tokens, max_tokens - generated_tokens),
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True
                )
                
                # 获取新生成的token
                new_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
                
                if len(new_tokens) == 0:
                    break
                
                # 解码并输出
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                if text:
                    yield text
                    self._stats["total_tokens"] += len(new_tokens)
                
                # 更新输入
                inputs = {'input_ids': outputs.sequences}
                generated_tokens += len(new_tokens)
                
                # 检查是否结束
                if outputs.sequences[0][-1].item() == self.tokenizer.eos_token_id:
                    break
                
                # 控制刷新率
                await asyncio.sleep(1.0 / self.config.refresh_rate)
    
    async def _retrieve_memory(self, query: str) -> List[Dict]:
        """检索记忆"""
        if not self.memory_system:
            return []
        
        try:
            results = await self.memory_system.retrieve(
                query,
                top_k=self.config.memory_top_k
            )
            return results
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return []
    
    async def _search_wiki(self, query: str) -> Optional[Dict]:
        """搜索维基百科"""
        if not self.tool_manager:
            return None
        
        try:
            result = await self.tool_manager.search_wikipedia(query)
            return result
        except Exception as e:
            logger.error(f"Wiki search error: {e}")
            return None
    
    def _build_reasoning_context(
        self,
        query: str,
        memory_results: Optional[List[Dict]] = None
    ) -> str:
        """构建推理上下文"""
        context_parts = [f"问题: {query}"]
        
        if memory_results:
            memory_context = "\n".join([
                f"- {m.get('content', '')}"
                for m in memory_results[:3]
            ])
            if memory_context:
                context_parts.append(f"\n相关记忆:\n{memory_context}")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "reasoning_history_count": len(self._reasoning_history)
        }


class ParallelProcessor:
    """
    并行处理器
    
    实现高刷新率的并行处理：
    - 并行记忆检索
    - 并行工具调用
    - 并行推理步骤
    """
    
    def __init__(self, refresh_rate: float = 30.0):
        self.refresh_rate = refresh_rate
        self._interval = 1.0 / refresh_rate
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._result_queue: asyncio.Queue = asyncio.Queue()
    
    async def submit_parallel(
        self,
        tasks: List[Callable]
    ) -> List[Any]:
        """
        并行提交任务
        
        在高刷新率循环中并行执行多个任务
        """
        results = await asyncio.gather(
            *[task() for task in tasks],
            return_exceptions=True
        )
        
        return [
            r if not isinstance(r, Exception) else None
            for r in results
        ]
    
    async def process_stream(
        self,
        input_stream: AsyncGenerator,
        processors: List[Callable]
    ) -> AsyncGenerator[Dict, None]:
        """
        流式处理输入
        
        对每个输入元素并行应用多个处理器
        """
        async for item in input_stream:
            start_time = time.time()
            
            # 并行处理
            results = await self.submit_parallel([
                lambda i=item, p=p: p(i) for p in processors
            ])
            
            # 输出结果
            yield {
                "input": item,
                "results": results,
                "elapsed": time.time() - start_time
            }
            
            # 控制刷新率
            elapsed = time.time() - start_time
            if elapsed < self._interval:
                await asyncio.sleep(self._interval - elapsed)


class StreamingBuffer:
    """
    流式缓冲区
    
    管理流式输出的缓冲和刷新
    """
    
    def __init__(self, max_size: int = 1000, flush_interval: float = 0.033):
        self.max_size = max_size
        self.flush_interval = flush_interval
        self._buffer: deque = deque(maxlen=max_size)
        self._last_flush = time.time()
    
    def append(self, item: Any) -> None:
        """添加项目"""
        self._buffer.append(item)
    
    def should_flush(self) -> bool:
        """是否应该刷新"""
        return (
            len(self._buffer) >= self.max_size or
            time.time() - self._last_flush >= self.flush_interval
        )
    
    def flush(self) -> List[Any]:
        """刷新缓冲区"""
        items = list(self._buffer)
        self._buffer.clear()
        self._last_flush = time.time()
        return items
