"""
海马体记忆系统
模拟人脑海马体的记忆编码、存储和检索功能
实现存算分离架构
"""

import asyncio
import json
import time
import sqlite3
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """记忆类型"""
    EPISODIC = "episodic"  # 情景记忆
    SEMANTIC = "semantic"  # 语义记忆
    PROCEDURAL = "procedural"  # 程序记忆
    WORKING = "working"  # 工作记忆


class MemoryDuration(Enum):
    """记忆持续时间"""
    SHORT_TERM = "short_term"  # 短期记忆 (< 30秒)
    LONG_TERM = "long_term"  # 长期记忆
    PERMANENT = "permanent"  # 永久记忆


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    capacity: int = 10000  # 记忆容量
    storage_backend: str = "sqlite"  # 存储后端
    storage_path: str = "./data/memory"
    
    # 时间参数
    short_term_duration: float = 30.0  # 短期记忆持续时间（秒）
    consolidation_threshold: int = 3  # 巩固阈值（访问次数）
    
    # 检索参数
    embedding_dim: int = 768
    similarity_threshold: float = 0.5
    max_retrieval_results: int = 10
    
    # 遗忘曲线参数
    forgetting_rate: float = 0.1
    rehearsal_bonus: float = 0.2


@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    content: str
    memory_type: str
    duration: str
    
    # 向量嵌入
    embedding: Optional[np.ndarray] = None
    
    # 元数据
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    
    # 关联
    associations: List[str] = field(default_factory=list)
    source: str = "user"
    
    # 巩固状态
    consolidated: bool = False
    importance: float = 0.5
    
    # 遗忘参数
    retention_strength: float = 1.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "duration": self.duration,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_access": self.last_access,
            "associations": self.associations,
            "source": self.source,
            "consolidated": self.consolidated,
            "importance": self.importance,
            "retention_strength": self.retention_strength
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        """从字典创建"""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=data["memory_type"],
            duration=data["duration"],
            timestamp=data.get("timestamp", time.time()),
            access_count=data.get("access_count", 0),
            last_access=data.get("last_access", time.time()),
            associations=data.get("associations", []),
            source=data.get("source", "user"),
            consolidated=data.get("consolidated", False),
            importance=data.get("importance", 0.5),
            retention_strength=data.get("retention_strength", 1.0)
        )


class HippocampusMemory:
    """
    海马体记忆系统
    
    模拟人脑海马体的功能：
    1. 记忆编码 - 将信息转换为记忆表示
    2. 记忆巩固 - 短期记忆转长期记忆
    3. 记忆检索 - 基于线索的联想检索
    4. 记忆遗忘 - 模拟遗忘曲线
    
    存算分离架构：
    - 存储层：独立的持久化存储
    - 计算层：轻量级内存索引
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        
        # 存储后端
        self._storage: Optional[MemoryStorage] = None
        
        # 内存索引（计算层）
        self._index: Dict[str, MemoryEntry] = {}
        self._embedding_index: Dict[str, np.ndarray] = {}
        
        # 神经累积增长追踪
        self._neural_growth: Dict[str, int] = {}
        
        # 统计
        self._stats = {
            "total_stored": 0,
            "total_retrieved": 0,
            "consolidated": 0,
            "forgotten": 0
        }
    
    async def initialize(self) -> None:
        """初始化记忆系统"""
        # 创建存储目录
        Path(self.config.storage_path).mkdir(parents=True, exist_ok=True)
        
        # 初始化存储后端
        self._storage = MemoryStorage(
            backend=self.config.storage_backend,
            path=self.config.storage_path
        )
        await self._storage.initialize()
        
        # 加载已有记忆到索引
        await self._load_index()
        
        logger.info(f"HippocampusMemory initialized with capacity {self.config.capacity}")
    
    async def _load_index(self) -> None:
        """加载记忆索引"""
        memories = await self._storage.load_all()
        
        for memory in memories:
            self._index[memory.id] = memory
            if memory.embedding is not None:
                self._embedding_index[memory.id] = memory.embedding
        
        logger.info(f"Loaded {len(self._index)} memories into index")
    
    async def store(
        self,
        entry: Dict[str, Any],
        embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        存储新记忆
        
        Args:
            entry: 记忆条目数据
            embedding: 可选的嵌入向量
            
        Returns:
            记忆ID
        """
        # 生成记忆ID
        memory_id = self._generate_id(entry)
        
        # 确定记忆类型和持续时间
        memory_type = entry.get("memory_type", MemoryType.EPISODIC.value)
        duration = entry.get("duration", MemoryDuration.SHORT_TERM.value)
        
        # 创建记忆条目
        memory = MemoryEntry(
            id=memory_id,
            content=entry.get("content", ""),
            memory_type=memory_type,
            duration=duration,
            embedding=embedding,
            timestamp=time.time(),
            source=entry.get("source", "user"),
            importance=entry.get("importance", 0.5)
        )
        
        # 计算嵌入（如果没有提供）
        if memory.embedding is None:
            memory.embedding = await self._compute_embedding(memory.content)
        
        # 存储到持久化层
        await self._storage.store(memory)
        
        # 更新索引
        self._index[memory_id] = memory
        if memory.embedding is not None:
            self._embedding_index[memory_id] = memory.embedding
        
        # 更新神经增长追踪
        self._update_neural_growth(memory)
        
        self._stats["total_stored"] += 1
        
        logger.debug(f"Stored memory: {memory_id}")
        return memory_id
    
    def _generate_id(self, entry: Dict) -> str:
        """生成唯一记忆ID"""
        content = entry.get("content", "")
        timestamp = time.time()
        hash_input = f"{content}{timestamp}".encode()
        return hashlib.md5(hash_input).hexdigest()[:16]
    
    async def _compute_embedding(self, text: str) -> np.ndarray:
        """计算文本嵌入"""
        # 简化的嵌入计算（实际应使用嵌入模型）
        # 使用简单的哈希向量化
        embedding = np.zeros(self.config.embedding_dim)
        
        words = text.lower().split()
        for word in words:
            word_hash = hash(word) % self.config.embedding_dim
            embedding[word_hash] += 1
        
        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _update_neural_growth(self, memory: MemoryEntry) -> None:
        """更新神经累积增长追踪"""
        # 模拟海马体神经元的累积增长
        # 新记忆会形成新的神经连接
        memory_type = memory.memory_type
        
        if memory_type not in self._neural_growth:
            self._neural_growth[memory_type] = 0
        
        self._neural_growth[memory_type] += 1
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        检索记忆
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            memory_type: 可选的记忆类型过滤
            min_similarity: 最小相似度阈值
            
        Returns:
            匹配的记忆列表
        """
        if not self._index:
            return []
        
        # 计算查询嵌入
        query_embedding = await self._compute_embedding(query)
        
        # 计算相似度
        similarities = []
        
        for memory_id, memory in self._index.items():
            # 类型过滤
            if memory_type and memory.memory_type != memory_type:
                continue
            
            if memory_id not in self._embedding_index:
                continue
            
            # 计算余弦相似度
            memory_embedding = self._embedding_index[memory_id]
            similarity = np.dot(query_embedding, memory_embedding)
            
            if similarity >= min_similarity:
                similarities.append((memory_id, similarity, memory))
        
        # 排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k结果
        results = []
        for memory_id, similarity, memory in similarities[:top_k]:
            # 更新访问信息
            memory.access_count += 1
            memory.last_access = time.time()
            
            # 检查是否需要巩固
            if self._should_consolidate(memory):
                await self._consolidate(memory)
            
            results.append({
                "id": memory_id,
                "content": memory.content,
                "similarity": float(similarity),
                "memory_type": memory.memory_type,
                "access_count": memory.access_count,
                "timestamp": memory.timestamp
            })
        
        self._stats["total_retrieved"] += len(results)
        
        return results
    
    def _should_consolidate(self, memory: MemoryEntry) -> bool:
        """判断是否应该巩固记忆"""
        if memory.consolidated:
            return False
        
        # 访问次数达到阈值
        if memory.access_count >= self.config.consolidation_threshold:
            return True
        
        # 重要性高
        if memory.importance > 0.8:
            return True
        
        return False
    
    async def _consolidate(self, memory: MemoryEntry) -> None:
        """巩固记忆（短期→长期）"""
        if memory.duration == MemoryDuration.SHORT_TERM.value:
            memory.duration = MemoryDuration.LONG_TERM.value
            memory.consolidated = True
            memory.retention_strength = 1.0
            
            # 更新存储
            await self._storage.update(memory)
            
            self._stats["consolidated"] += 1
            logger.debug(f"Consolidated memory: {memory.id}")
    
    async def forget(self, force: bool = False) -> int:
        """
        应用遗忘机制
        
        模拟人脑的遗忘曲线，移除不再需要的记忆
        
        Args:
            force: 是否强制遗忘（忽略重要性）
            
        Returns:
            遗忘的记忆数量
        """
        current_time = time.time()
        forgotten_count = 0
        
        to_forget = []
        
        for memory_id, memory in self._index.items():
            # 永久记忆不遗忘
            if memory.duration == MemoryDuration.PERMANENT.value:
                continue
            
            # 计算遗忘概率
            time_since_access = current_time - memory.last_access
            forgetting_prob = self._compute_forgetting_probability(
                time_since_access,
                memory.access_count,
                memory.importance
            )
            
            if force or np.random.random() < forgetting_prob:
                to_forget.append(memory_id)
        
        # 执行遗忘
        for memory_id in to_forget:
            await self._storage.delete(memory_id)
            del self._index[memory_id]
            if memory_id in self._embedding_index:
                del self._embedding_index[memory_id]
            forgotten_count += 1
        
        self._stats["forgotten"] += forgotten_count
        
        if forgotten_count > 0:
            logger.info(f"Forgot {forgotten_count} memories")
        
        return forgotten_count
    
    def _compute_forgetting_probability(
        self,
        time_since_access: float,
        access_count: int,
        importance: float
    ) -> float:
        """
        计算遗忘概率
        
        基于艾宾浩斯遗忘曲线：
        R = e^(-t/S)
        其中 R 是保持率，t 是时间，S 是记忆强度
        """
        # 记忆强度受访问次数和重要性影响
        strength = (
            self.config.forgetting_rate *
            (1 + access_count * 0.1) *
            (1 + importance)
        )
        
        # 保持率
        retention = np.exp(-time_since_access / (strength * 3600))
        
        # 遗忘概率
        forgetting_prob = 1 - retention
        
        return min(1.0, max(0.0, forgetting_prob))
    
    async def associate(
        self,
        memory_id_1: str,
        memory_id_2: str,
        association_strength: float = 1.0
    ) -> None:
        """
        建立记忆关联
        
        模拟人脑的联想记忆机制
        """
        if memory_id_1 not in self._index or memory_id_2 not in self._index:
            return
        
        memory1 = self._index[memory_id_1]
        memory2 = self._index[memory_id_2]
        
        # 双向关联
        if memory_id_2 not in memory1.associations:
            memory1.associations.append(memory_id_2)
        
        if memory_id_1 not in memory2.associations:
            memory2.associations.append(memory_id_1)
        
        # 更新存储
        await self._storage.update(memory1)
        await self._storage.update(memory2)
    
    async def get_associated_memories(
        self,
        memory_id: str,
        depth: int = 1
    ) -> List[Dict]:
        """
        获取关联记忆
        
        Args:
            memory_id: 起始记忆ID
            depth: 搜索深度
            
        Returns:
            关联的记忆列表
        """
        if memory_id not in self._index:
            return []
        
        visited = set()
        result = []
        queue = [(memory_id, 0)]
        
        while queue:
            current_id, current_depth = queue.pop(0)
            
            if current_id in visited or current_depth > depth:
                continue
            
            visited.add(current_id)
            
            if current_id in self._index:
                memory = self._index[current_id]
                result.append({
                    "id": memory.id,
                    "content": memory.content,
                    "depth": current_depth
                })
                
                # 添加关联记忆到队列
                for assoc_id in memory.associations:
                    if assoc_id not in visited:
                        queue.append((assoc_id, current_depth + 1))
        
        return result[1:]  # 排除起始记忆
    
    def get_neural_growth_stats(self) -> Dict[str, int]:
        """获取神经累积增长统计"""
        return dict(self._neural_growth)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计"""
        return {
            **self._stats,
            "total_memories": len(self._index),
            "memory_types": self._count_by_type(),
            "neural_growth": self.get_neural_growth_stats()
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """按类型统计记忆"""
        counts = {}
        for memory in self._index.values():
            mtype = memory.memory_type
            counts[mtype] = counts.get(mtype, 0) + 1
        return counts
    
    async def close(self) -> None:
        """关闭记忆系统"""
        if self._storage:
            await self._storage.close()
        
        logger.info("HippocampusMemory closed")


class MemoryStorage:
    """
    记忆存储后端
    
    实现存算分离的存储层
    """
    
    def __init__(
        self,
        backend: str = "sqlite",
        path: str = "./data/memory"
    ):
        self.backend = backend
        self.path = Path(path)
        self._db: Optional[sqlite3.Connection] = None
    
    async def initialize(self) -> None:
        """初始化存储"""
        self.path.mkdir(parents=True, exist_ok=True)
        
        if self.backend == "sqlite":
            db_path = self.path / "memory.db"
            self._db = sqlite3.connect(str(db_path))
            
            # 创建表
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    memory_type TEXT,
                    duration TEXT,
                    timestamp REAL,
                    access_count INTEGER,
                    last_access REAL,
                    associations TEXT,
                    source TEXT,
                    consolidated INTEGER,
                    importance REAL,
                    retention_strength REAL,
                    embedding BLOB
                )
            """)
            self._db.commit()
    
    async def store(self, memory: MemoryEntry) -> None:
        """存储记忆"""
        if self._db is None:
            return
        
        embedding_blob = None
        if memory.embedding is not None:
            embedding_blob = memory.embedding.tobytes()
        
        self._db.execute("""
            INSERT OR REPLACE INTO memories VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            memory.id,
            memory.content,
            memory.memory_type,
            memory.duration,
            memory.timestamp,
            memory.access_count,
            memory.last_access,
            json.dumps(memory.associations),
            memory.source,
            int(memory.consolidated),
            memory.importance,
            memory.retention_strength,
            embedding_blob
        ))
        self._db.commit()
    
    async def load(self, memory_id: str) -> Optional[MemoryEntry]:
        """加载单个记忆"""
        if self._db is None:
            return None
        
        cursor = self._db.execute(
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_memory(row)
    
    async def load_all(self) -> List[MemoryEntry]:
        """加载所有记忆"""
        if self._db is None:
            return []
        
        cursor = self._db.execute("SELECT * FROM memories")
        rows = cursor.fetchall()
        
        return [self._row_to_memory(row) for row in rows]
    
    def _row_to_memory(self, row: tuple) -> MemoryEntry:
        """将数据库行转换为记忆对象"""
        memory = MemoryEntry(
            id=row[0],
            content=row[1],
            memory_type=row[2],
            duration=row[3],
            timestamp=row[4],
            access_count=row[5],
            last_access=row[6],
            associations=json.loads(row[7]) if row[7] else [],
            source=row[8],
            consolidated=bool(row[9]),
            importance=row[10],
            retention_strength=row[11]
        )
        
        # 恢复嵌入
        if row[12]:
            memory.embedding = np.frombuffer(row[12], dtype=np.float64)
        
        return memory
    
    async def update(self, memory: MemoryEntry) -> None:
        """更新记忆"""
        await self.store(memory)
    
    async def delete(self, memory_id: str) -> None:
        """删除记忆"""
        if self._db is None:
            return
        
        self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._db.commit()
    
    async def close(self) -> None:
        """关闭存储"""
        if self._db:
            self._db.close()
