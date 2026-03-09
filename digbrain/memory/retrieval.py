"""
记忆检索模块
实现高效的记忆搜索和检索
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging
import time
import heapq

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """检索配置"""
    embedding_dim: int = 768
    similarity_threshold: float = 0.3
    max_results: int = 10
    
    # 索引配置
    use_approximate: bool = True  # 使用近似最近邻
    n_clusters: int = 100  # 聚类数量
    
    # 时间衰减
    time_decay_factor: float = 0.1
    
    # 重要性加权
    importance_weight: float = 0.2


class MemoryRetriever:
    """
    记忆检索器
    
    实现多种检索策略：
    1. 向量相似度检索
    2. 时间衰减检索
    3. 重要性加权检索
    4. 联想检索
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        
        # 索引
        self._embeddings: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict] = {}
        
        # 近似索引
        self._cluster_centers: Optional[np.ndarray] = None
        self._cluster_assignments: Dict[int, List[str]] = {}
        
        # 统计
        self._stats = {
            "total_queries": 0,
            "avg_query_time": 0.0,
            "cache_hits": 0
        }
    
    def index_memory(
        self,
        memory_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        索引记忆
        
        Args:
            memory_id: 记忆ID
            embedding: 嵌入向量
            metadata: 元数据
        """
        # 归一化嵌入
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        self._embeddings[memory_id] = embedding
        self._metadata[memory_id] = metadata or {}
    
    def remove_memory(self, memory_id: str) -> None:
        """移除记忆索引"""
        self._embeddings.pop(memory_id, None)
        self._metadata.pop(memory_id, None)
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        检索相关记忆
        
        Args:
            query_embedding: 查询嵌入
            top_k: 返回数量
            filters: 过滤条件
            
        Returns:
            (memory_id, score, metadata) 列表
        """
        start_time = time.time()
        self._stats["total_queries"] += 1
        
        # 归一化查询
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        if self.config.use_approximate and self._cluster_centers is not None:
            results = self._approximate_search(query_embedding, top_k, filters)
        else:
            results = self._exact_search(query_embedding, top_k, filters)
        
        # 更新统计
        query_time = time.time() - start_time
        self._stats["avg_query_time"] = (
            self._stats["avg_query_time"] * (self._stats["total_queries"] - 1) +
            query_time
        ) / self._stats["total_queries"]
        
        return results
    
    def _exact_search(
        self,
        query: np.ndarray,
        top_k: int,
        filters: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        """精确搜索"""
        scores = []
        
        for memory_id, embedding in self._embeddings.items():
            # 应用过滤
            if filters and not self._match_filters(memory_id, filters):
                continue
            
            # 计算相似度
            similarity = np.dot(query, embedding)
            
            # 应用时间衰减
            metadata = self._metadata.get(memory_id, {})
            time_decay = self._compute_time_decay(metadata)
            
            # 应用重要性加权
            importance = metadata.get("importance", 0.5)
            weighted_score = (
                similarity * (1 - self.config.importance_weight) +
                importance * self.config.importance_weight
            ) * time_decay
            
            scores.append((memory_id, weighted_score, metadata))
        
        # 排序并返回top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _approximate_search(
        self,
        query: np.ndarray,
        top_k: int,
        filters: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        """近似搜索（基于聚类）"""
        # 找到最近的聚类中心
        cluster_similarities = np.dot(self._cluster_centers, query)
        nearest_cluster = np.argmax(cluster_similarities)
        
        # 在最近的聚类中搜索
        candidates = self._cluster_assignments.get(nearest_cluster, [])
        
        scores = []
        for memory_id in candidates:
            if memory_id not in self._embeddings:
                continue
            
            if filters and not self._match_filters(memory_id, filters):
                continue
            
            embedding = self._embeddings[memory_id]
            similarity = np.dot(query, embedding)
            
            metadata = self._metadata.get(memory_id, {})
            time_decay = self._compute_time_decay(metadata)
            importance = metadata.get("importance", 0.5)
            
            weighted_score = (
                similarity * (1 - self.config.importance_weight) +
                importance * self.config.importance_weight
            ) * time_decay
            
            scores.append((memory_id, weighted_score, metadata))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _match_filters(self, memory_id: str, filters: Dict) -> bool:
        """检查是否匹配过滤条件"""
        metadata = self._metadata.get(memory_id, {})
        
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        
        return True
    
    def _compute_time_decay(self, metadata: Dict) -> float:
        """计算时间衰减因子"""
        if "timestamp" not in metadata:
            return 1.0
        
        age = time.time() - metadata["timestamp"]
        decay = np.exp(-self.config.time_decay_factor * age / 86400)  # 天为单位
        return decay
    
    def build_index(self) -> None:
        """构建近似索引"""
        if len(self._embeddings) < self.config.n_clusters:
            logger.warning("Not enough memories for clustering")
            return
        
        # 简单的K-means聚类
        embeddings_matrix = np.array(list(self._embeddings.values()))
        memory_ids = list(self._embeddings.keys())
        
        # 初始化聚类中心
        indices = np.random.choice(
            len(embeddings_matrix),
            self.config.n_clusters,
            replace=False
        )
        self._cluster_centers = embeddings_matrix[indices].copy()
        
        # 迭代聚类
        for _ in range(10):  # 最多10次迭代
            # 分配
            assignments = np.argmax(
                np.dot(embeddings_matrix, self._cluster_centers.T),
                axis=1
            )
            
            # 更新中心
            new_centers = np.zeros_like(self._cluster_centers)
            counts = np.zeros(self.config.n_clusters)
            
            for i, cluster_id in enumerate(assignments):
                new_centers[cluster_id] += embeddings_matrix[i]
                counts[cluster_id] += 1
            
            for j in range(self.config.n_clusters):
                if counts[j] > 0:
                    new_centers[j] /= counts[j]
                else:
                    new_centers[j] = self._cluster_centers[j]
            
            # 检查收敛
            if np.allclose(self._cluster_centers, new_centers):
                break
            
            self._cluster_centers = new_centers
        
        # 构建分配映射
        self._cluster_assignments = {i: [] for i in range(self.config.n_clusters)}
        
        for i, memory_id in enumerate(memory_ids):
            cluster_id = np.argmax(
                np.dot(embeddings_matrix[i], self._cluster_centers.T)
            )
            self._cluster_assignments[cluster_id].append(memory_id)
        
        logger.info(f"Built index with {self.config.n_clusters} clusters")
    
    def retrieve_by_association(
        self,
        memory_id: str,
        association_strength: float = 0.5,
        max_depth: int = 2
    ) -> List[Tuple[str, float, Dict]]:
        """
        联想检索
        
        基于记忆关联进行检索
        """
        results = []
        visited = set()
        queue = [(memory_id, 1.0, 0)]
        
        while queue:
            current_id, current_strength, depth = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            if current_id in self._metadata:
                metadata = self._metadata[current_id]
                results.append((current_id, current_strength, metadata))
                
                # 添加关联记忆
                associations = metadata.get("associations", [])
                for assoc_id in associations:
                    if assoc_id not in visited:
                        new_strength = current_strength * association_strength
                        queue.append((assoc_id, new_strength, depth + 1))
        
        return results[1:]  # 排除起始记忆
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "indexed_memories": len(self._embeddings),
            "clusters": len(self._cluster_assignments)
        }


class HybridRetriever:
    """
    混合检索器
    
    结合多种检索策略
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.vector_retriever = MemoryRetriever(self.config)
    
    def index_memory(
        self,
        memory_id: str,
        embedding: np.ndarray,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """索引记忆"""
        enhanced_metadata = metadata or {}
        enhanced_metadata["content"] = content
        self.vector_retriever.index_memory(memory_id, embedding, enhanced_metadata)
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
        mode: str = "hybrid"
    ) -> List[Tuple[str, float, Dict]]:
        """
        混合检索
        
        Args:
            query_embedding: 查询嵌入
            query_text: 查询文本
            top_k: 返回数量
            mode: 检索模式 (vector, keyword, hybrid)
            
        Returns:
            检索结果
        """
        if mode == "vector":
            return self.vector_retriever.retrieve(query_embedding, top_k)
        elif mode == "keyword":
            return self._keyword_search(query_text, top_k)
        else:  # hybrid
            vector_results = self.vector_retriever.retrieve(query_embedding, top_k * 2)
            keyword_results = self._keyword_search(query_text, top_k * 2)
            
            return self._merge_results(vector_results, keyword_results, top_k)
    
    def _keyword_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[str, float, Dict]]:
        """关键词搜索"""
        query_words = set(query.lower().split())
        scores = []
        
        for memory_id, metadata in self.vector_retriever._metadata.items():
            content = metadata.get("content", "").lower()
            content_words = set(content.split())
            
            # Jaccard相似度
            intersection = len(query_words & content_words)
            union = len(query_words | content_words)
            
            if union > 0:
                similarity = intersection / union
                scores.append((memory_id, similarity, metadata))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _merge_results(
        self,
        vector_results: List[Tuple[str, float, Dict]],
        keyword_results: List[Tuple[str, float, Dict]],
        top_k: int
    ) -> List[Tuple[str, float, Dict]]:
        """合并检索结果"""
        # 使用倒数排名融合
        scores = {}
        
        for rank, (memory_id, _, metadata) in enumerate(vector_results):
            scores[memory_id] = scores.get(memory_id, 0) + 1 / (rank + 60)
        
        for rank, (memory_id, _, metadata) in enumerate(keyword_results):
            scores[memory_id] = scores.get(memory_id, 0) + 1 / (rank + 60)
        
        # 排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        results = []
        for memory_id in sorted_ids[:top_k]:
            metadata = self.vector_retriever._metadata.get(memory_id, {})
            results.append((memory_id, scores[memory_id], metadata))
        
        return results
