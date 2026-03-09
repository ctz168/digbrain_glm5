"""
记忆存储模块
实现存算分离的存储后端
"""

import json
import sqlite3
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """存储后端类型"""
    SQLITE = "sqlite"
    FILE = "file"
    REDIS = "redis"
    MEMORY = "memory"


@dataclass
class StorageConfig:
    """存储配置"""
    backend: StorageBackend = StorageBackend.SQLITE
    path: str = "./data/memory"
    cache_size: int = 1000
    async_writes: bool = True


class MemoryStorage:
    """
    记忆存储类
    
    支持多种存储后端，实现存算分离
    """
    
    def __init__(
        self,
        backend: str = "sqlite",
        path: str = "./data/memory",
        config: Optional[StorageConfig] = None
    ):
        self.config = config or StorageConfig()
        self.backend = StorageBackend(backend)
        self.path = Path(path)
        
        # 连接/资源
        self._db: Optional[sqlite3.Connection] = None
        self._cache: Dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # 统计
        self._stats = {
            "reads": 0,
            "writes": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def initialize(self) -> None:
        """初始化存储"""
        self.path.mkdir(parents=True, exist_ok=True)
        
        if self.backend == StorageBackend.SQLITE:
            await self._init_sqlite()
        elif self.backend == StorageBackend.FILE:
            await self._init_file()
        elif self.backend == StorageBackend.MEMORY:
            self._cache = {}
        
        logger.info(f"MemoryStorage initialized with {self.backend.value} backend")
    
    async def _init_sqlite(self) -> None:
        """初始化SQLite存储"""
        db_path = self.path / "memory.db"
        
        # 在线程池中执行阻塞操作
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._create_sqlite_db,
            str(db_path)
        )
    
    def _create_sqlite_db(self, db_path: str) -> None:
        """创建SQLite数据库"""
        self._db = sqlite3.connect(db_path)
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
                embedding BLOB,
                metadata TEXT
            )
        """)
        
        # 创建索引
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
        """)
        self._db.commit()
    
    async def _init_file(self) -> None:
        """初始化文件存储"""
        memories_dir = self.path / "memories"
        memories_dir.mkdir(parents=True, exist_ok=True)
    
    async def store(self, memory: Any) -> None:
        """存储记忆"""
        self._stats["writes"] += 1
        
        if self.backend == StorageBackend.SQLITE:
            await self._store_sqlite(memory)
        elif self.backend == StorageBackend.FILE:
            await self._store_file(memory)
        elif self.backend == StorageBackend.MEMORY:
            self._cache[memory.id] = memory
    
    async def _store_sqlite(self, memory: Any) -> None:
        """SQLite存储"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._write_sqlite,
            memory
        )
    
    def _write_sqlite(self, memory: Any) -> None:
        """写入SQLite"""
        if self._db is None:
            return
        
        embedding_blob = None
        if hasattr(memory, 'embedding') and memory.embedding is not None:
            embedding_blob = memory.embedding.tobytes()
        
        associations = "[]"
        if hasattr(memory, 'associations'):
            associations = json.dumps(memory.associations)
        
        metadata = "{}"
        if hasattr(memory, 'metadata'):
            metadata = json.dumps(memory.metadata)
        
        self._db.execute("""
            INSERT OR REPLACE INTO memories VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            memory.id,
            memory.content,
            memory.memory_type,
            memory.duration,
            memory.timestamp,
            memory.access_count,
            memory.last_access,
            associations,
            memory.source,
            int(memory.consolidated),
            memory.importance,
            memory.retention_strength,
            embedding_blob,
            metadata
        ))
        self._db.commit()
    
    async def _store_file(self, memory: Any) -> None:
        """文件存储"""
        memory_file = self.path / "memories" / f"{memory.id}.json"
        
        data = {
            "id": memory.id,
            "content": memory.content,
            "memory_type": memory.memory_type,
            "duration": memory.duration,
            "timestamp": memory.timestamp,
            "access_count": memory.access_count,
            "last_access": memory.last_access,
            "associations": memory.associations,
            "source": memory.source,
            "consolidated": memory.consolidated,
            "importance": memory.importance,
            "retention_strength": memory.retention_strength
        }
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._write_file,
            str(memory_file),
            data
        )
    
    def _write_file(self, filepath: str, data: Dict) -> None:
        """写入文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    async def load(self, memory_id: str) -> Optional[Any]:
        """加载记忆"""
        self._stats["reads"] += 1
        
        # 检查缓存
        if memory_id in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[memory_id]
        
        self._stats["cache_misses"] += 1
        
        if self.backend == StorageBackend.SQLITE:
            return await self._load_sqlite(memory_id)
        elif self.backend == StorageBackend.FILE:
            return await self._load_file(memory_id)
        elif self.backend == StorageBackend.MEMORY:
            return self._cache.get(memory_id)
        
        return None
    
    async def _load_sqlite(self, memory_id: str) -> Optional[Any]:
        """从SQLite加载"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._read_sqlite,
            memory_id
        )
    
    def _read_sqlite(self, memory_id: str) -> Optional[Any]:
        """读取SQLite"""
        if self._db is None:
            return None
        
        cursor = self._db.execute(
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        # 转换为字典
        from .hippocampus import MemoryEntry
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
        
        if row[12]:
            memory.embedding = np.frombuffer(row[12], dtype=np.float64)
        
        return memory
    
    async def _load_file(self, memory_id: str) -> Optional[Any]:
        """从文件加载"""
        memory_file = self.path / "memories" / f"{memory_id}.json"
        
        if not memory_file.exists():
            return None
        
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            self._executor,
            self._read_file,
            str(memory_file)
        )
        
        if data:
            from .hippocampus import MemoryEntry
            return MemoryEntry.from_dict(data)
        
        return None
    
    def _read_file(self, filepath: str) -> Optional[Dict]:
        """读取文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read file {filepath}: {e}")
            return None
    
    async def load_all(self) -> List[Any]:
        """加载所有记忆"""
        if self.backend == StorageBackend.SQLITE:
            return await self._load_all_sqlite()
        elif self.backend == StorageBackend.FILE:
            return await self._load_all_file()
        elif self.backend == StorageBackend.MEMORY:
            return list(self._cache.values())
        
        return []
    
    async def _load_all_sqlite(self) -> List[Any]:
        """从SQLite加载所有"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._read_all_sqlite
        )
    
    def _read_all_sqlite(self) -> List[Any]:
        """读取所有SQLite记录"""
        if self._db is None:
            return []
        
        cursor = self._db.execute("SELECT * FROM memories")
        rows = cursor.fetchall()
        
        memories = []
        from .hippocampus import MemoryEntry
        
        for row in rows:
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
            
            if row[12]:
                memory.embedding = np.frombuffer(row[12], dtype=np.float64)
            
            memories.append(memory)
        
        return memories
    
    async def _load_all_file(self) -> List[Any]:
        """从文件加载所有"""
        memories_dir = self.path / "memories"
        memories = []
        
        for memory_file in memories_dir.glob("*.json"):
            memory = await self._load_file(memory_file.stem)
            if memory:
                memories.append(memory)
        
        return memories
    
    async def update(self, memory: Any) -> None:
        """更新记忆"""
        await self.store(memory)
    
    async def delete(self, memory_id: str) -> None:
        """删除记忆"""
        if self.backend == StorageBackend.SQLITE:
            await self._delete_sqlite(memory_id)
        elif self.backend == StorageBackend.FILE:
            await self._delete_file(memory_id)
        elif self.backend == StorageBackend.MEMORY:
            self._cache.pop(memory_id, None)
    
    async def _delete_sqlite(self, memory_id: str) -> None:
        """从SQLite删除"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._delete_sqlite_sync,
            memory_id
        )
    
    def _delete_sqlite_sync(self, memory_id: str) -> None:
        """同步删除SQLite记录"""
        if self._db:
            self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._db.commit()
    
    async def _delete_file(self, memory_id: str) -> None:
        """删除文件"""
        memory_file = self.path / "memories" / f"{memory_id}.json"
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            memory_file.unlink,
            True
        )
    
    async def close(self) -> None:
        """关闭存储"""
        if self._db:
            self._db.close()
        
        self._executor.shutdown(wait=True)
        
        logger.info("MemoryStorage closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "backend": self.backend.value,
            "cache_size": len(self._cache)
        }
