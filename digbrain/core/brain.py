"""
DigBrain - 类脑智能系统核心模块
实现高刷新率流式处理、存算分离、在线STDP学习
"""

import asyncio
import time
import json
import os
from typing import Optional, Dict, Any, List, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """处理模式枚举"""
    STREAMING = "streaming"  # 流式处理
    BATCH = "batch"  # 批量处理
    INTERACTIVE = "interactive"  # 交互模式


class MemoryType(Enum):
    """记忆类型枚举"""
    SHORT_TERM = "short_term"  # 短期记忆
    LONG_TERM = "long_term"  # 长期记忆
    PERMANENT = "permanent"  # 永久记忆


@dataclass
class BrainConfig:
    """大脑配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # 使用可用的Qwen模型
    model_path: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # 流式处理配置
    refresh_rate: float = 30.0  # Hz, 处理刷新率
    chunk_size: int = 64  # 每次处理的token数量
    max_context_length: int = 4096  # 最大上下文长度
    
    # 记忆配置
    memory_capacity: int = 10000  # 记忆容量
    short_term_duration: float = 30.0  # 短期记忆持续时间（秒）
    long_term_threshold: float = 3.0  # 转入长期记忆的访问阈值
    
    # STDP学习配置
    stdp_enabled: bool = True  # 是否启用STDP学习
    stdp_learning_rate: float = 0.01  # STDP学习率
    stdp_window: float = 20.0  # STDP时间窗口（毫秒）
    
    # 存算分离配置
    storage_backend: str = "sqlite"  # sqlite, redis, file
    storage_path: str = "./data/memory"
    
    # 多模态配置
    enable_vision: bool = True  # 启用视觉处理
    image_size: tuple = (224, 224)  # 图像处理尺寸
    
    # 工具配置
    enable_wiki_search: bool = True  # 启用维基百科搜索
    enable_web_tools: bool = True  # 启用网页工具
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@dataclass
class ProcessingContext:
    """处理上下文"""
    session_id: str
    input_buffer: List[str] = field(default_factory=list)
    output_buffer: List[str] = field(default_factory=list)
    memory_queries: List[Dict] = field(default_factory=list)
    attention_weights: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DigBrain:
    """
    DigBrain类脑智能系统主类
    
    实现核心功能：
    1. 高刷新率流式处理
    2. 存算分离架构
    3. 在线STDP学习
    4. 多模态处理
    5. 类人脑记忆管理
    """
    
    def __init__(self, config: Optional[BrainConfig] = None):
        """
        初始化DigBrain系统
        
        Args:
            config: 大脑配置，如果为None则使用默认配置
        """
        self.config = config or BrainConfig()
        self._initialized = False
        self._model = None
        self._tokenizer = None
        self._memory_system = None
        self._stdp_engine = None
        self._stream_processor = None
        self._tool_manager = None
        
        # 运行时状态
        self._processing = False
        self._contexts: Dict[str, ProcessingContext] = {}
        self._neuron_weights: Dict[str, np.ndarray] = {}
        
        # 统计信息
        self._stats = {
            "total_processed": 0,
            "total_tokens": 0,
            "memory_queries": 0,
            "wiki_searches": 0,
            "stdp_updates": 0
        }
        
        logger.info(f"DigBrain initialized with config: {self.config}")
    
    async def initialize(self) -> None:
        """
        异步初始化所有组件
        
        加载模型、初始化记忆系统、启动流处理器
        """
        if self._initialized:
            return
            
        logger.info("Initializing DigBrain components...")
        
        # 初始化模型
        await self._init_model()
        
        # 初始化记忆系统
        await self._init_memory()
        
        # 初始化STDP引擎
        await self._init_stdp()
        
        # 初始化流处理器
        await self._init_stream_processor()
        
        # 初始化工具管理器
        await self._init_tools()
        
        self._initialized = True
        logger.info("DigBrain initialization complete")
    
    async def _init_model(self) -> None:
        """初始化Qwen模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
            import torch
            
            model_path = self.config.model_path or self.config.model_name
            logger.info(f"Loading model from: {model_path}")
            
            # 确定设备
            if self.config.device == "auto":
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            else:
                device = self.config.device
            
            logger.info(f"Using device: {device}")
            
            # 加载tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 加载模型
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device,
                trust_remote_code=True
            )
            
            # 如果有视觉能力，加载processor
            try:
                self._processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
            except:
                self._processor = None
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # 使用回退方案
            self._model = None
            self._tokenizer = None
            logger.warning("Running in fallback mode without local model")
    
    async def _init_memory(self) -> None:
        """初始化记忆系统"""
        from ..memory import HippocampusMemory, MemoryConfig
        
        memory_config = MemoryConfig(
            capacity=self.config.memory_capacity,
            storage_backend=self.config.storage_backend,
            storage_path=self.config.storage_path,
            short_term_duration=self.config.short_term_duration,
            long_term_threshold=self.config.long_term_threshold
        )
        
        self._memory_system = HippocampusMemory(memory_config)
        await self._memory_system.initialize()
        
        logger.info("Memory system initialized")
    
    async def _init_stdp(self) -> None:
        """初始化STDP学习引擎"""
        if not self.config.stdp_enabled:
            return
            
        from ..training import STDPEngine, STDPConfig
        
        stdp_config = STDPConfig(
            learning_rate=self.config.stdp_learning_rate,
            time_window=self.config.stdp_window
        )
        
        self._stdp_engine = STDPEngine(stdp_config)
        await self._stdp_engine.initialize()
        
        logger.info("STDP engine initialized")
    
    async def _init_stream_processor(self) -> None:
        """初始化流处理器"""
        from .stream import StreamProcessor, StreamConfig
        
        stream_config = StreamConfig(
            refresh_rate=self.config.refresh_rate,
            chunk_size=self.config.chunk_size,
            max_context_length=self.config.max_context_length
        )
        
        self._stream_processor = StreamProcessor(stream_config)
        await self._stream_processor.initialize()
        
        logger.info("Stream processor initialized")
    
    async def _init_tools(self) -> None:
        """初始化工具管理器"""
        from ..tools import ToolManager, ToolConfig
        
        tool_config = ToolConfig(
            enable_wiki=self.config.enable_wiki_search,
            enable_web=self.config.enable_web_tools
        )
        
        self._tool_manager = ToolManager(tool_config)
        await self._tool_manager.initialize()
        
        logger.info("Tool manager initialized")
    
    async def process(
        self,
        input_data: str,
        session_id: Optional[str] = None,
        stream: bool = True,
        search_memory: bool = True,
        search_wiki: bool = False,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        处理输入数据（流式）
        
        Args:
            input_data: 输入文本
            session_id: 会话ID
            stream: 是否流式输出
            search_memory: 是否搜索记忆
            search_wiki: 是否搜索维基百科
            **kwargs: 其他参数
            
        Yields:
            处理结果（流式）
        """
        if not self._initialized:
            await self.initialize()
        
        # 创建或获取上下文
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())
        
        if session_id not in self._contexts:
            self._contexts[session_id] = ProcessingContext(session_id=session_id)
        
        context = self._contexts[session_id]
        context.input_buffer.append(input_data)
        
        # 高刷新率流式处理
        async for chunk in self._stream_process(
            input_data,
            context,
            search_memory=search_memory,
            search_wiki=search_wiki,
            **kwargs
        ):
            context.output_buffer.append(chunk)
            yield chunk
        
        # STDP在线学习更新
        if self.config.stdp_enabled and self._stdp_engine:
            await self._update_stdp(context)
        
        self._stats["total_processed"] += 1
    
    async def _stream_process(
        self,
        input_data: str,
        context: ProcessingContext,
        search_memory: bool = True,
        search_wiki: bool = False,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        内部流式处理方法
        
        实现高刷新率的并行处理：
        1. 并行搜索记忆和外部知识
        2. 流式推理生成
        3. 实时记忆存储
        """
        # 并行执行记忆检索和维基搜索
        memory_result = None
        wiki_result = None
        
        tasks = []
        
        if search_memory and self._memory_system:
            tasks.append(self._search_memory(input_data, context))
        
        if search_wiki and self._tool_manager:
            tasks.append(self._search_wiki(input_data))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Parallel task failed: {result}")
                elif i == 0 and search_memory:
                    memory_result = result
                elif i == 1 or (i == 0 and not search_memory):
                    wiki_result = result
        
        # 构建增强上下文
        enhanced_input = self._build_enhanced_input(
            input_data,
            memory_result,
            wiki_result
        )
        
        # 流式生成响应
        async for chunk in self._generate_stream(enhanced_input, context):
            yield chunk
        
        # 存储新记忆
        if self._memory_system:
            await self._store_memory(input_data, context.output_buffer, context)
    
    async def _search_memory(
        self,
        query: str,
        context: ProcessingContext
    ) -> Optional[List[Dict]]:
        """搜索记忆系统"""
        if not self._memory_system:
            return None
        
        self._stats["memory_queries"] += 1
        
        try:
            results = await self._memory_system.retrieve(query, top_k=5)
            context.memory_queries.append({
                "query": query,
                "results": results,
                "timestamp": time.time()
            })
            return results
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return None
    
    async def _search_wiki(self, query: str) -> Optional[Dict]:
        """搜索维基百科"""
        if not self._tool_manager:
            return None
        
        self._stats["wiki_searches"] += 1
        
        try:
            result = await self._tool_manager.search_wikipedia(query)
            return result
        except Exception as e:
            logger.error(f"Wiki search failed: {e}")
            return None
    
    def _build_enhanced_input(
        self,
        input_data: str,
        memory_result: Optional[List[Dict]],
        wiki_result: Optional[Dict]
    ) -> str:
        """构建增强输入"""
        parts = []
        
        # 添加记忆上下文
        if memory_result:
            memory_context = "\n".join([
                f"[记忆] {m.get('content', '')}"
                for m in memory_result[:3]
            ])
            if memory_context:
                parts.append(f"相关记忆:\n{memory_context}\n")
        
        # 添加维基百科知识
        if wiki_result:
            wiki_context = wiki_result.get("summary", "")
            if wiki_context:
                parts.append(f"知识库:\n{wiki_context}\n")
        
        # 添加原始输入
        parts.append(f"用户输入: {input_data}")
        
        return "\n".join(parts)
    
    async def _generate_stream(
        self,
        input_data: str,
        context: ProcessingContext
    ) -> AsyncGenerator[str, None]:
        """流式生成响应"""
        if self._model and self._tokenizer:
            # 使用本地模型生成
            async for chunk in self._generate_with_model(input_data, context):
                yield chunk
        else:
            # 回退到简单响应
            yield self._fallback_response(input_data)
    
    async def _generate_with_model(
        self,
        input_data: str,
        context: ProcessingContext
    ) -> AsyncGenerator[str, None]:
        """使用模型生成流式响应"""
        import torch
        
        # 编码输入
        inputs = self._tokenizer(
            input_data,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        # 流式生成
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                stream=True
            )
            
            for output in outputs:
                chunk = self._tokenizer.decode(
                    output[inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                if chunk:
                    yield chunk
                    self._stats["total_tokens"] += 1
    
    def _fallback_response(self, input_data: str) -> str:
        """回退响应（无模型时）"""
        return f"[DigBrain] 收到输入: {input_data[:100]}... (模型未加载，请检查模型路径)"
    
    async def _store_memory(
        self,
        input_data: str,
        output_buffer: List[str],
        context: ProcessingContext
    ) -> None:
        """存储记忆"""
        if not self._memory_system:
            return
        
        memory_entry = {
            "input": input_data,
            "output": "".join(output_buffer),
            "timestamp": time.time(),
            "session_id": context.session_id,
            "memory_type": MemoryType.SHORT_TERM.value
        }
        
        await self._memory_system.store(memory_entry)
    
    async def _update_stdp(self, context: ProcessingContext) -> None:
        """更新STDP权重"""
        if not self._stdp_engine:
            return
        
        self._stats["stdp_updates"] += 1
        
        # 计算时间差并更新权重
        input_time = context.timestamp
        output_time = time.time()
        
        await self._stdp_engine.update(
            pre_spike_time=input_time,
            post_spike_time=output_time,
            context=context
        )
    
    async def process_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        video: Optional[str] = None,
        session_id: Optional[str] = None,
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        多模态处理
        
        Args:
            text: 文本输入
            image: 图像路径或base64
            video: 视频路径
            session_id: 会话ID
            stream: 是否流式输出
            
        Yields:
            处理结果
        """
        if not self._initialized:
            await self.initialize()
        
        # 处理图像
        if image:
            image_content = await self._process_image(image)
            if text:
                text = f"[图像描述: {image_content}]\n{text}"
            else:
                text = f"请描述这张图片: [图像已加载]"
        
        # 处理视频（逐帧）
        if video:
            frames = await self._extract_video_frames(video)
            for i, frame in enumerate(frames):
                frame_desc = await self._process_image(frame)
                async for chunk in self.process(
                    f"[视频帧{i+1}] {frame_desc}",
                    session_id=session_id,
                    stream=stream,
                    **kwargs
                ):
                    yield chunk
            return
        
        # 处理文本
        if text:
            async for chunk in self.process(
                text,
                session_id=session_id,
                stream=stream,
                **kwargs
            ):
                yield chunk
    
    async def _process_image(self, image_path: str) -> str:
        """处理图像"""
        try:
            from PIL import Image
            import base64
            import io
            
            # 加载图像
            if image_path.startswith("data:image"):
                # Base64图像
                image_data = base64.b64decode(image_path.split(",")[1])
                image = Image.open(io.BytesIO(image_data))
            else:
                # 文件路径
                image = Image.open(image_path)
            
            # 如果有视觉模型，使用它
            if self._processor and hasattr(self._model, 'vision_tower'):
                # 多模态模型处理
                inputs = self._processor(images=image, return_tensors="pt")
                # ... 处理逻辑
                return "[图像已处理]"
            else:
                # 简单描述
                return f"[图像: {image.size[0]}x{image.size[1]}]"
                
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return "[图像处理失败]"
    
    async def _extract_video_frames(self, video_path: str) -> List[str]:
        """提取视频帧"""
        try:
            import cv2
            import tempfile
            
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            # 每秒提取1帧
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps) if fps > 0 else 1
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # 保存帧到临时文件
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                        cv2.imwrite(f.name, frame)
                        frames.append(f.name)
                
                frame_count += 1
                
                # 限制帧数
                if len(frames) >= 30:
                    break
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Video frame extraction failed: {e}")
            return []
    
    def enable_online_learning(self, stdp_rate: float = 0.01) -> None:
        """启用在线学习"""
        self.config.stdp_enabled = True
        self.config.stdp_learning_rate = stdp_rate
        logger.info(f"Online learning enabled with STDP rate: {stdp_rate}")
    
    def disable_online_learning(self) -> None:
        """禁用在线学习"""
        self.config.stdp_enabled = False
        logger.info("Online learning disabled")
    
    async def save_weights(self, path: str) -> None:
        """保存权重"""
        weights_path = Path(path)
        weights_path.mkdir(parents=True, exist_ok=True)
        
        # 保存神经元权重
        for name, weights in self._neuron_weights.items():
            np.save(weights_path / f"{name}.npy", weights)
        
        # 保存STDP状态
        if self._stdp_engine:
            await self._stdp_engine.save_state(weights_path / "stdp_state.json")
        
        logger.info(f"Weights saved to {path}")
    
    async def load_weights(self, path: str) -> None:
        """加载权重"""
        weights_path = Path(path)
        
        # 加载神经元权重
        for weight_file in weights_path.glob("*.npy"):
            name = weight_file.stem
            self._neuron_weights[name] = np.load(weight_file)
        
        # 加载STDP状态
        if self._stdp_engine and (weights_path / "stdp_state.json").exists():
            await self._stdp_engine.load_state(weights_path / "stdp_state.json")
        
        logger.info(f"Weights loaded from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "initialized": self._initialized,
            "model_loaded": self._model is not None,
            "memory_active": self._memory_system is not None,
            "stdp_active": self._stdp_engine is not None,
            "active_sessions": len(self._contexts)
        }
    
    async def shutdown(self) -> None:
        """关闭系统"""
        logger.info("Shutting down DigBrain...")
        
        if self._memory_system:
            await self._memory_system.close()
        
        self._initialized = False
        logger.info("DigBrain shutdown complete")


# 便捷函数
async def create_brain(config: Optional[BrainConfig] = None) -> DigBrain:
    """创建并初始化DigBrain实例"""
    brain = DigBrain(config)
    await brain.initialize()
    return brain
