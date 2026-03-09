"""
评估指标模块
实现各种评估指标计算
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    BLEU = "bleu"
    ROUGE = "rouge"
    PERPLEXITY = "perplexity"
    LATENCY = "latency"


@dataclass
class MetricResult:
    """指标结果"""
    metric_type: str
    value: float
    details: Optional[Dict] = None


class MetricsCalculator:
    """
    评估指标计算器
    
    支持多种评估指标：
    - 分类指标：准确率、精确率、召回率、F1
    - 生成指标：BLEU、ROUGE
    - 语言模型指标：困惑度
    - 系统指标：延迟、吞吐量
    """
    
    def __init__(self):
        self._results: Dict[str, List[MetricResult]] = {}
    
    def calculate_accuracy(
        self,
        predictions: List[Any],
        labels: List[Any]
    ) -> MetricResult:
        """计算准确率"""
        correct = sum(p == l for p, l in zip(predictions, labels))
        accuracy = correct / len(labels) if labels else 0
        
        return MetricResult(
            metric_type=MetricType.ACCURACY.value,
            value=accuracy,
            details={"correct": correct, "total": len(labels)}
        )
    
    def calculate_precision_recall_f1(
        self,
        predictions: List[Any],
        labels: List[Any],
        average: str = "macro"
    ) -> Tuple[MetricResult, MetricResult, MetricResult]:
        """计算精确率、召回率、F1"""
        # 获取所有类别
        classes = set(predictions) | set(labels)
        
        precisions = []
        recalls = []
        f1s = []
        
        for cls in classes:
            tp = sum(1 for p, l in zip(predictions, labels) if p == cls and l == cls)
            fp = sum(1 for p, l in zip(predictions, labels) if p == cls and l != cls)
            fn = sum(1 for p, l in zip(predictions, labels) if p != cls and l == cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        if average == "macro":
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = np.mean(f1s)
        else:  # micro
            total_tp = sum(1 for p, l in zip(predictions, labels) if p == l)
            precision = total_tp / len(predictions) if predictions else 0
            recall = total_tp / len(labels) if labels else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return (
            MetricResult(MetricType.PRECISION.value, precision),
            MetricResult(MetricType.RECALL.value, recall),
            MetricResult(MetricType.F1.value, f1)
        )
    
    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[List[str]],
        max_n: int = 4
    ) -> MetricResult:
        """
        计算BLEU分数
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表（每个预测可以有多个参考）
            max_n: 最大n-gram
        """
        bleu_scores = []
        
        for pred, refs in zip(predictions, references):
            # 确保refs是列表
            if isinstance(refs, str):
                refs = [refs]
            
            pred_tokens = pred.split()
            
            # 计算各阶n-gram精度
            precisions = []
            for n in range(1, max_n + 1):
                pred_ngrams = self._get_ngrams(pred_tokens, n)
                
                max_counts = {}
                for ref in refs:
                    ref_tokens = ref.split()
                    ref_ngrams = self._get_ngrams(ref_tokens, n)
                    
                    for ngram, count in ref_ngrams.items():
                        max_counts[ngram] = max(max_counts.get(ngram, 0), count)
                
                clipped_count = 0
                total_count = 0
                
                for ngram, count in pred_ngrams.items():
                    clipped_count += min(count, max_counts.get(ngram, 0))
                    total_count += count
                
                precision = clipped_count / total_count if total_count > 0 else 0
                precisions.append(precision)
            
            # 计算简短惩罚
            ref_lengths = [len(ref.split()) for ref in refs]
            closest_ref_len = min(ref_lengths, key=lambda x: abs(x - len(pred_tokens)))
            
            if len(pred_tokens) > closest_ref_len:
                bp = 1
            else:
                bp = np.exp(1 - closest_ref_len / len(pred_tokens)) if len(pred_tokens) > 0 else 0
            
            # 计算BLEU
            if all(p > 0 for p in precisions):
                bleu = bp * np.exp(np.mean(np.log(precisions)))
            else:
                bleu = 0
            
            bleu_scores.append(bleu)
        
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
        
        return MetricResult(
            metric_type=MetricType.BLEU.value,
            value=avg_bleu,
            details={"individual_scores": bleu_scores}
        )
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """获取n-gram"""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] += 1
        return ngrams
    
    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str],
        rouge_type: str = "rouge-l"
    ) -> MetricResult:
        """
        计算ROUGE分数
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            rouge_type: ROUGE类型 (rouge-1, rouge-2, rouge-l)
        """
        rouge_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            if rouge_type == "rouge-l":
                # 最长公共子序列
                lcs_len = self._lcs_length(pred_tokens, ref_tokens)
                
                precision = lcs_len / len(pred_tokens) if pred_tokens else 0
                recall = lcs_len / len(ref_tokens) if ref_tokens else 0
                
            else:
                # n-gram重叠
                n = int(rouge_type.split("-")[1])
                pred_ngrams = set(self._get_ngrams(pred_tokens, n).keys())
                ref_ngrams = set(self._get_ngrams(ref_tokens, n).keys())
                
                overlap = len(pred_ngrams & ref_ngrams)
                
                precision = overlap / len(pred_ngrams) if pred_ngrams else 0
                recall = overlap / len(ref_ngrams) if ref_ngrams else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            rouge_scores.append(f1)
        
        avg_rouge = np.mean(rouge_scores) if rouge_scores else 0
        
        return MetricResult(
            metric_type=f"{MetricType.ROUGE.value}_{rouge_type}",
            value=avg_rouge,
            details={"individual_scores": rouge_scores}
        )
    
    def _lcs_length(self, seq1: List, seq2: List) -> int:
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def calculate_perplexity(
        self,
        log_probs: List[float]
    ) -> MetricResult:
        """
        计算困惑度
        
        Args:
            log_probs: 对数概率列表
        """
        if not log_probs:
            return MetricResult(MetricType.PERPLEXITY.value, float('inf'))
        
        avg_log_prob = np.mean(log_probs)
        perplexity = np.exp(-avg_log_prob)
        
        return MetricResult(
            metric_type=MetricType.PERPLEXITY.value,
            value=perplexity,
            details={"avg_log_prob": avg_log_prob}
        )
    
    def calculate_latency_stats(
        self,
        latencies: List[float]
    ) -> MetricResult:
        """计算延迟统计"""
        if not latencies:
            return MetricResult(MetricType.LATENCY.value, 0)
        
        return MetricResult(
            metric_type=MetricType.LATENCY.value,
            value=np.mean(latencies),
            details={
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "std": np.std(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99)
            }
        )
    
    def compute_all_classification_metrics(
        self,
        predictions: List[Any],
        labels: List[Any]
    ) -> Dict[str, MetricResult]:
        """计算所有分类指标"""
        accuracy = self.calculate_accuracy(predictions, labels)
        precision, recall, f1 = self.calculate_precision_recall_f1(predictions, labels)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def compute_all_generation_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, MetricResult]:
        """计算所有生成指标"""
        bleu = self.calculate_bleu(predictions, [[r] for r in references])
        rouge_1 = self.calculate_rouge(predictions, references, "rouge-1")
        rouge_2 = self.calculate_rouge(predictions, references, "rouge-2")
        rouge_l = self.calculate_rouge(predictions, references, "rouge-l")
        
        return {
            "bleu": bleu,
            "rouge_1": rouge_1,
            "rouge_2": rouge_2,
            "rouge_l": rouge_l
        }
    
    def add_result(self, name: str, result: MetricResult) -> None:
        """添加结果"""
        if name not in self._results:
            self._results[name] = []
        self._results[name].append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取汇总"""
        summary = {}
        
        for name, results in self._results.items():
            values = [r.value for r in results]
            summary[name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "count": len(values)
            }
        
        return summary
