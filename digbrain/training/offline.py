"""
离线训练模块
实现批量训练和多线程训练
"""

import asyncio
import time
import json
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import numpy as np
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@dataclass
class OfflineConfig:
    """离线训练配置"""
    # 基础配置
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # 学习率调度
    lr_scheduler: str = "cosine"  # cosine, linear, exponential
    warmup_epochs: int = 1
    min_lr: float = 1e-6
    
    # 正则化
    weight_decay: float = 0.01
    dropout: float = 0.1
    gradient_clip: float = 1.0
    
    # 多线程
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # 检查点
    save_every: int = 1
    checkpoint_dir: str = "./checkpoints"
    
    # 早停
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.001


class OfflineTrainer:
    """
    离线训练器
    
    支持：
    - 批量训练
    - 多线程数据加载
    - 学习率调度
    - 检查点保存
    - 早停机制
    """
    
    def __init__(self, config: Optional[OfflineConfig] = None):
        self.config = config or OfflineConfig()
        
        # 训练状态
        self._current_epoch = 0
        self._current_lr = self.config.learning_rate
        self._best_loss = float('inf')
        self._patience_counter = 0
        
        # 数据加载
        self._executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        
        # 统计
        self._stats = {
            "total_epochs": 0,
            "total_steps": 0,
            "train_loss_history": [],
            "val_loss_history": [],
            "lr_history": []
        }
        
        # 锁
        self._lock = threading.Lock()
    
    async def train(
        self,
        model: Any,
        train_data: List[Any],
        val_data: Optional[List[Any]] = None,
        loss_fn: Optional[Callable] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            model: 要训练的模型
            train_data: 训练数据
            val_data: 验证数据
            loss_fn: 损失函数
            callbacks: 回调函数列表
            
        Returns:
            训练结果
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        # 创建检查点目录
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config.epochs):
            self._current_epoch = epoch
            epoch_start = time.time()
            
            # 学习率调度
            self._update_learning_rate(epoch)
            
            # 训练一个epoch
            train_loss = await self._train_epoch(
                model, train_data, loss_fn
            )
            
            # 验证
            val_loss = None
            if val_data:
                val_loss = await self._validate(model, val_data, loss_fn)
            
            # 记录统计
            self._stats["total_epochs"] += 1
            self._stats["train_loss_history"].append(train_loss)
            if val_loss is not None:
                self._stats["val_loss_history"].append(val_loss)
            self._stats["lr_history"].append(self._current_lr)
            
            # 回调
            if callbacks:
                for callback in callbacks:
                    await callback(epoch, train_loss, val_loss, model)
            
            # 保存检查点
            if (epoch + 1) % self.config.save_every == 0:
                await self._save_checkpoint(model, epoch)
            
            # 早停检查
            if self.config.early_stopping and val_loss is not None:
                if self._check_early_stopping(val_loss):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f if val_loss else 'N/A'}, "
                f"lr={self._current_lr:.6f}, time={epoch_time:.2f}s"
            )
        
        return {
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "total_epochs": self._current_epoch + 1,
            "best_loss": self._best_loss,
            "stats": self._stats
        }
    
    async def _train_epoch(
        self,
        model: Any,
        data: List[Any],
        loss_fn: Optional[Callable]
    ) -> float:
        """训练一个epoch"""
        total_loss = 0.0
        num_batches = 0
        
        # 批处理
        for i in range(0, len(data), self.config.batch_size):
            batch = data[i:i + self.config.batch_size]
            
            # 训练一个batch
            loss = await self._train_batch(model, batch, loss_fn)
            total_loss += loss
            num_batches += 1
            
            self._stats["total_steps"] += 1
        
        return total_loss / max(num_batches, 1)
    
    async def _train_batch(
        self,
        model: Any,
        batch: List[Any],
        loss_fn: Optional[Callable]
    ) -> float:
        """训练一个batch"""
        # 简化实现
        return np.random.random() * 0.1
    
    async def _validate(
        self,
        model: Any,
        data: List[Any],
        loss_fn: Optional[Callable]
    ) -> float:
        """验证"""
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(data), self.config.batch_size):
            batch = data[i:i + self.config.batch_size]
            loss = await self._train_batch(model, batch, loss_fn)
            total_loss += loss
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _update_learning_rate(self, epoch: int) -> None:
        """更新学习率"""
        if self.config.lr_scheduler == "cosine":
            # 余弦退火
            progress = epoch / self.config.epochs
            self._current_lr = self.config.min_lr + 0.5 * (
                self.config.learning_rate - self.config.min_lr
            ) * (1 + np.cos(np.pi * progress))
        
        elif self.config.lr_scheduler == "linear":
            # 线性衰减
            progress = epoch / self.config.epochs
            self._current_lr = self.config.learning_rate * (1 - progress)
        
        elif self.config.lr_scheduler == "exponential":
            # 指数衰减
            self._current_lr = self.config.learning_rate * (0.9 ** epoch)
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """检查早停"""
        if val_loss < self._best_loss - self.config.min_delta:
            self._best_loss = val_loss
            self._patience_counter = 0
            return False
        else:
            self._patience_counter += 1
            return self._patience_counter >= self.config.patience
    
    async def _save_checkpoint(self, model: Any, epoch: int) -> None:
        """保存检查点"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_{epoch}.pt"
        
        checkpoint = {
            "epoch": epoch,
            "learning_rate": self._current_lr,
            "best_loss": self._best_loss,
            "stats": self._stats
        }
        
        # 保存模型权重（简化）
        # 实际应保存model.state_dict()
        
        with open(checkpoint_path.with_suffix(".json"), 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    async def load_checkpoint(self, model: Any, checkpoint_path: str) -> None:
        """加载检查点"""
        path = Path(checkpoint_path)
        
        with open(path.with_suffix(".json"), 'r') as f:
            checkpoint = json.load(f)
        
        self._current_epoch = checkpoint["epoch"]
        self._current_lr = checkpoint["learning_rate"]
        self._best_loss = checkpoint["best_loss"]
        self._stats = checkpoint["stats"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


class MultiThreadTrainer(OfflineTrainer):
    """
    多线程训练器
    
    支持并行数据加载和训练
    """
    
    def __init__(self, config: Optional[OfflineConfig] = None):
        super().__init__(config)
        
        # 数据队列
        self._data_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.prefetch_factor * self.config.num_workers
        )
        
        # 工作线程
        self._workers: List[threading.Thread] = []
    
    async def train(
        self,
        model: Any,
        train_data: List[Any],
        val_data: Optional[List[Any]] = None,
        loss_fn: Optional[Callable] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """多线程训练"""
        # 启动数据加载线程
        self._start_workers(train_data)
        
        try:
            result = await super().train(model, train_data, val_data, loss_fn, callbacks)
        finally:
            # 停止工作线程
            self._stop_workers()
        
        return result
    
    def _start_workers(self, data: List[Any]) -> None:
        """启动工作线程"""
        for i in range(self.config.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(data, i),
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
    
    def _stop_workers(self) -> None:
        """停止工作线程"""
        for _ in self._workers:
            try:
                self._data_queue.put_nowait(None)  # 停止信号
            except asyncio.QueueFull:
                pass
        
        for worker in self._workers:
            worker.join(timeout=5)
        
        self._workers.clear()
    
    def _worker_loop(self, data: List[Any], worker_id: int) -> None:
        """工作线程循环"""
        logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                # 获取数据
                idx = np.random.randint(len(data))
                batch = data[idx:idx + self.config.batch_size]
                
                # 预处理
                processed = self._preprocess_batch(batch)
                
                # 放入队列
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._data_queue.put(processed))
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                break
        
        logger.info(f"Worker {worker_id} stopped")
    
    def _preprocess_batch(self, batch: List[Any]) -> Any:
        """预处理batch"""
        return batch


class ModuleTrainer:
    """
    模块训练器
    
    支持单独训练各个模块
    """
    
    def __init__(self):
        self._trainers: Dict[str, OfflineTrainer] = {}
    
    def register_module(
        self,
        module_name: str,
        config: Optional[OfflineConfig] = None
    ) -> None:
        """注册模块"""
        self._trainers[module_name] = OfflineTrainer(config)
    
    async def train_module(
        self,
        module_name: str,
        model: Any,
        data: List[Any],
        **kwargs
    ) -> Dict[str, Any]:
        """训练单个模块"""
        if module_name not in self._trainers:
            raise ValueError(f"Module {module_name} not registered")
        
        return await self._trainers[module_name].train(model, data, **kwargs)
    
    async def train_all(
        self,
        models: Dict[str, Any],
        data: Dict[str, List[Any]],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """训练所有模块"""
        results = {}
        
        for module_name, model in models.items():
            if module_name in data:
                results[module_name] = await self.train_module(
                    module_name, model, data[module_name], **kwargs
                )
        
        return results
    
    async def parallel_train(
        self,
        models: Dict[str, Any],
        data: Dict[str, List[Any]],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """并行训练所有模块"""
        tasks = []
        
        for module_name, model in models.items():
            if module_name in data and module_name in self._trainers:
                task = self.train_module(module_name, model, data[module_name], **kwargs)
                tasks.append((module_name, task))
        
        results = {}
        for module_name, task in tasks:
            results[module_name] = await task
        
        return results
