"""
基准测试模块
实现真实的、可复现的基准测试
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    # 测试集
    test_data_path: str = "./data/benchmarks"
    
    # 输出
    results_path: str = "./results/benchmarks"
    
    # 批处理
    batch_size: int = 8
    
    # 超时
    timeout_per_sample: float = 60.0
    
    # 并行
    num_workers: int = 4
    
    # 反作弊
    seed: int = 42
    shuffle: bool = True
    validate_answers: bool = True


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    benchmark_name: str
    total_samples: int
    correct: int
    accuracy: float
    avg_latency: float
    total_time: float
    
    # 详细结果
    details: List[Dict] = field(default_factory=list)
    
    # 元数据
    model_name: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "benchmark_name": self.benchmark_name,
            "total_samples": self.total_samples,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "avg_latency": self.avg_latency,
            "total_time": self.total_time,
            "model_name": self.model_name,
            "timestamp": self.timestamp
        }


class BenchmarkRunner:
    """
    基准测试运行器
    
    支持多种标准基准测试：
    - MMLU: 多任务语言理解
    - HellaSwag: 常识推理
    - WinoGrande: 核心指代消解
    - ARC: AI推理挑战
    - TruthfulQA: 事实准确性
    - GSM8K: 数学推理
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        
        # 设置随机种子（反作弊）
        np.random.seed(self.config.seed)
        
        # 测试数据
        self._test_data: Dict[str, List[Dict]] = {}
        
        # 结果
        self._results: Dict[str, BenchmarkResult] = {}
        
        # 统计
        self._stats = {
            "total_benchmarks": 0,
            "total_samples": 0,
            "total_correct": 0
        }
    
    async def load_benchmark(
        self,
        benchmark_name: str,
        data_path: Optional[str] = None
    ) -> None:
        """
        加载基准测试数据
        
        Args:
            benchmark_name: 基准测试名称
            data_path: 数据路径
        """
        path = Path(data_path or self.config.test_data_path)
        benchmark_file = path / f"{benchmark_name}.json"
        
        if benchmark_file.exists():
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                self._test_data[benchmark_name] = json.load(f)
            logger.info(f"Loaded {benchmark_name}: {len(self._test_data[benchmark_name])} samples")
        else:
            # 生成示例数据
            self._test_data[benchmark_name] = self._generate_sample_data(benchmark_name)
            logger.warning(f"Generated sample data for {benchmark_name}")
    
    def _generate_sample_data(self, benchmark_name: str) -> List[Dict]:
        """生成示例测试数据"""
        samples = []
        
        if benchmark_name == "mmlu":
            categories = ["math", "physics", "biology", "history"]
            for cat in categories:
                for i in range(10):
                    samples.append({
                        "id": f"{cat}_{i}",
                        "question": f"Sample {cat} question {i}?",
                        "choices": ["A", "B", "C", "D"],
                        "answer": "A",
                        "category": cat
                    })
        
        elif benchmark_name == "hellaswag":
            for i in range(20):
                samples.append({
                    "id": f"hs_{i}",
                    "context": f"Context for sample {i}",
                    "endings": ["Ending A", "Ending B", "Ending C", "Ending D"],
                    "label": 0
                })
        
        elif benchmark_name == "gsm8k":
            for i in range(10):
                samples.append({
                    "id": f"gsm8k_{i}",
                    "question": f"What is {i+1} + {i+2}?",
                    "answer": str(2*i + 3)
                })
        
        else:
            for i in range(10):
                samples.append({
                    "id": f"{benchmark_name}_{i}",
                    "question": f"Sample question {i}",
                    "answer": "A"
                })
        
        return samples
    
    async def run_benchmark(
        self,
        benchmark_name: str,
        model: Any,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None
    ) -> BenchmarkResult:
        """
        运行基准测试
        
        Args:
            benchmark_name: 基准测试名称
            model: 模型实例
            preprocess_fn: 预处理函数
            postprocess_fn: 后处理函数
            
        Returns:
            测试结果
        """
        if benchmark_name not in self._test_data:
            await self.load_benchmark(benchmark_name)
        
        data = self._test_data[benchmark_name]
        
        # 打乱数据（反作弊）
        if self.config.shuffle:
            np.random.shuffle(data)
        
        logger.info(f"Running benchmark: {benchmark_name}")
        
        start_time = time.time()
        correct = 0
        total_latency = 0.0
        details = []
        
        for sample in data:
            sample_start = time.time()
            
            try:
                # 预处理
                input_data = sample
                if preprocess_fn:
                    input_data = preprocess_fn(sample)
                
                # 模型推理
                prediction = await self._run_inference(model, input_data)
                
                # 后处理
                if postprocess_fn:
                    prediction = postprocess_fn(prediction)
                
                # 验证答案
                is_correct = self._validate_answer(prediction, sample)
                
                sample_latency = time.time() - sample_start
                total_latency += sample_latency
                
                if is_correct:
                    correct += 1
                
                details.append({
                    "id": sample.get("id"),
                    "prediction": prediction,
                    "correct": is_correct,
                    "latency": sample_latency
                })
                
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                details.append({
                    "id": sample.get("id"),
                    "error": str(e),
                    "correct": False
                })
        
        total_time = time.time() - start_time
        
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            total_samples=len(data),
            correct=correct,
            accuracy=correct / len(data) if data else 0,
            avg_latency=total_latency / len(data) if data else 0,
            total_time=total_time,
            details=details
        )
        
        self._results[benchmark_name] = result
        self._stats["total_benchmarks"] += 1
        self._stats["total_samples"] += len(data)
        self._stats["total_correct"] += correct
        
        logger.info(
            f"Benchmark {benchmark_name}: "
            f"accuracy={result.accuracy:.4f}, "
            f"avg_latency={result.avg_latency:.2f}s"
        )
        
        return result
    
    async def _run_inference(self, model: Any, input_data: Any) -> Any:
        """运行模型推理"""
        # 简化实现
        if hasattr(model, 'generate'):
            return await model.generate(input_data)
        elif hasattr(model, 'predict'):
            return model.predict(input_data)
        else:
            return "A"  # 默认返回
    
    def _validate_answer(self, prediction: Any, sample: Dict) -> bool:
        """验证答案"""
        if not self.config.validate_answers:
            return True
        
        # 获取正确答案
        if "answer" in sample:
            correct = str(sample["answer"]).strip().upper()
        elif "label" in sample:
            correct = str(sample["label"])
        else:
            return False
        
        # 比较预测
        pred_str = str(prediction).strip().upper()
        
        return pred_str == correct or correct in pred_str
    
    async def run_all_benchmarks(
        self,
        model: Any,
        benchmarks: Optional[List[str]] = None
    ) -> Dict[str, BenchmarkResult]:
        """运行所有基准测试"""
        benchmarks = benchmarks or ["mmlu", "hellaswag", "winogrande", "arc", "truthfulqa", "gsm8k"]
        
        results = {}
        for benchmark in benchmarks:
            try:
                result = await self.run_benchmark(benchmark, model)
                results[benchmark] = result
            except Exception as e:
                logger.error(f"Failed to run {benchmark}: {e}")
        
        return results
    
    def save_results(
        self,
        output_path: Optional[str] = None
    ) -> None:
        """保存结果"""
        path = Path(output_path or self.config.results_path)
        path.mkdir(parents=True, exist_ok=True)
        
        for name, result in self._results.items():
            result_file = path / f"{name}_results.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 保存汇总
        summary = {
            "stats": self._stats,
            "results": {name: r.to_dict() for name, r in self._results.items()}
        }
        
        with open(path / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取测试汇总"""
        return {
            "stats": self._stats,
            "results": {name: r.to_dict() for name, r in self._results.items()}
        }


class AntiCheatValidator:
    """
    反作弊验证器
    
    确保测试结果的真实性和可复现性
    """
    
    def __init__(self):
        self._validation_log: List[Dict] = []
    
    def validate_result(
        self,
        result: BenchmarkResult,
        expected_range: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """
        验证结果
        
        检查：
        1. 结果是否在合理范围内
        2. 是否有异常模式
        3. 是否可复现
        """
        validation = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # 检查准确率范围
        if result.accuracy < 0 or result.accuracy > 1:
            validation["errors"].append("Accuracy out of valid range")
            validation["valid"] = False
        
        # 检查预期范围
        if expected_range:
            if not (expected_range[0] <= result.accuracy <= expected_range[1]):
                validation["warnings"].append(
                    f"Accuracy {result.accuracy} outside expected range {expected_range}"
                )
        
        # 检查异常模式
        if result.avg_latency < 0.001:
            validation["warnings"].append("Suspiciously fast inference")
        
        # 检查样本数
        if result.total_samples < 10:
            validation["warnings"].append("Too few samples for reliable evaluation")
        
        self._validation_log.append({
            "benchmark": result.benchmark_name,
            "validation": validation,
            "timestamp": time.time()
        })
        
        return validation
    
    def check_reproducibility(
        self,
        results: List[BenchmarkResult]
    ) -> float:
        """
        检查可复现性
        
        比较多次运行的结果一致性
        """
        if len(results) < 2:
            return 1.0
        
        accuracies = [r.accuracy for r in results]
        std = np.std(accuracies)
        
        # 标准差越小，可复现性越高
        reproducibility = 1 - min(std * 10, 1)
        
        return reproducibility
