"""
流式处理引擎
实现高刷新率的流式输入输出处理
"""

import asyncio
import time
import queue
from typing import Optional, Dict, Any, AsyncGenerator, Callable, List
from dataclasses import dataclass
from enum import Enum
import logging
import threading

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """流状态"""
    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    CLOSED = "closed"


@dataclass
class StreamConfig:
    """流处理配置"""
    refresh_rate: float = 30.0  # Hz
    chunk_size: int = 64  # tokens per chunk
    max_context_length: int = 4096
    buffer_size: int = 1024
    timeout: float = 30.0


class StreamProcessor:
    """
    高刷新率流式处理器
    
    实现类似人脑的实时流式处理：
    - 高刷新率处理（可配置10-100Hz）
    - 小批量数据处理
    - 实时输入输出
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self._state = StreamState.IDLE
        self._input_queue: asyncio.Queue = asyncio.Queue()
        self._output_queue: asyncio.Queue = asyncio.Queue()
        self._buffer: List[str] = []
        self._processing_task: Optional[asyncio.Task] = None
        self._callbacks: Dict[str, Callable] = {}
        
        # 时间控制
        self._interval = 1.0 / self.config.refresh_rate
        self._last_process_time = 0.0
        
        # 统计
        self._stats = {
            "chunks_processed": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "avg_latency": 0.0
        }
    
    async def initialize(self) -> None:
        """初始化流处理器"""
        self._state = StreamState.IDLE
        logger.info(f"StreamProcessor initialized at {self.config.refresh_rate}Hz")
    
    async def start(self) -> None:
        """启动流处理"""
        if self._state == StreamState.PROCESSING:
            return
        
        self._state = StreamState.PROCESSING
        self._processing_task = asyncio.create_task(self._processing_loop())
        logger.info("Stream processing started")
    
    async def stop(self) -> None:
        """停止流处理"""
        self._state = StreamState.IDLE
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Stream processing stopped")
    
    async def _processing_loop(self) -> None:
        """主处理循环"""
        while self._state == StreamState.PROCESSING:
            start_time = time.time()
            
            try:
                # 尝试获取输入
                try:
                    input_chunk = await asyncio.wait_for(
                        self._input_queue.get(),
                        timeout=self._interval
                    )
                    
                    # 处理输入块
                    await self._process_chunk(input_chunk)
                    
                except asyncio.TimeoutError:
                    # 超时，继续循环
                    pass
                
                # 控制刷新率
                elapsed = time.time() - start_time
                if elapsed < self._interval:
                    await asyncio.sleep(self._interval - elapsed)
                    
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
    
    async def _process_chunk(self, chunk: Dict[str, Any]) -> None:
        """处理单个数据块"""
        self._stats["chunks_processed"] += 1
        
        # 更新统计
        if "tokens" in chunk:
            self._stats["total_input_tokens"] += chunk["tokens"]
        
        # 调用回调
        if "callback_id" in chunk and chunk["callback_id"] in self._callbacks:
            callback = self._callbacks[chunk["callback_id"]]
            result = await callback(chunk)
            await self._output_queue.put(result)
    
    async def submit(
        self,
        data: str,
        callback: Optional[Callable] = None,
        priority: int = 0
    ) -> str:
        """
        提交数据处理请求
        
        Args:
            data: 输入数据
            callback: 处理回调
            priority: 优先级
            
        Returns:
            处理结果ID
        """
        import uuid
        request_id = str(uuid.uuid4())
        
        chunk = {
            "id": request_id,
            "data": data,
            "tokens": len(data.split()),  # 简单估计
            "priority": priority,
            "timestamp": time.time()
        }
        
        if callback:
            self._callbacks[request_id] = callback
            chunk["callback_id"] = request_id
        
        await self._input_queue.put(chunk)
        return request_id
    
    async def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[Dict]:
        """获取处理结果"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = await asyncio.wait_for(
                    self._output_queue.get(),
                    timeout=1.0
                )
                if result.get("id") == request_id:
                    return result
                else:
                    # 放回队列
                    await self._output_queue.put(result)
            except asyncio.TimeoutError:
                continue
        
        return None
    
    def register_callback(self, callback_id: str, callback: Callable) -> None:
        """注册回调函数"""
        self._callbacks[callback_id] = callback
    
    def unregister_callback(self, callback_id: str) -> None:
        """注销回调函数"""
        self._callbacks.pop(callback_id, None)
    
    @property
    def state(self) -> StreamState:
        return self._state
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "state": self._state.value,
            "queue_size": self._input_queue.qsize(),
            "refresh_rate": self.config.refresh_rate
        }


class ChunkProcessor:
    """
    数据块处理器
    将大块数据分割成小块进行流式处理
    """
    
    def __init__(self, chunk_size: int = 64):
        self.chunk_size = chunk_size
    
    def chunk_text(self, text: str) -> List[str]:
        """将文本分割成块"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def chunk_tokens(self, tokens: List[int]) -> List[List[int]]:
        """将token列表分割成块"""
        return [
            tokens[i:i + self.chunk_size]
            for i in range(0, len(tokens), self.chunk_size)
        ]
    
    async def stream_chunks(
        self,
        data: str,
        processor: Callable[[str], AsyncGenerator[str, None]]
    ) -> AsyncGenerator[str, None]:
        """流式处理数据块"""
        chunks = self.chunk_text(data)
        
        for chunk in chunks:
            async for result in processor(chunk):
                yield result
