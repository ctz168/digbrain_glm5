"""
自适应推理模块
根据问题复杂度自动选择最优推理方法
"""

import re
import time
import asyncio
from typing import Optional, Dict, Any, List, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """复杂度级别"""
    SIMPLE = "simple"        # 简单问题 - 直接回答
    MODERATE = "moderate"    # 中等问题 - 简单推理
    COMPLEX = "complex"      # 复杂问题 - 深度推理
    VERY_COMPLEX = "very_complex"  # 非常复杂 - 多步推理


class ReasoningMethod(Enum):
    """推理方法"""
    DIRECT = "direct"              # 直接回答
    SIMPLE_COT = "simple_cot"      # 简单思维链
    DEEP_COT = "deep_cot"          # 深度思维链
    MULTI_STEP = "multi_step"      # 多步推理
    TOOL_ASSISTED = "tool_assisted"  # 工具辅助


@dataclass
class ComplexityFeatures:
    """复杂度特征"""
    # 文本特征
    word_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    
    # 问题特征
    has_multiple_questions: bool = False
    has_conditional: bool = False
    has_comparison: bool = False
    has_calculation: bool = False
    has_reasoning_keywords: bool = False
    
    # 领域特征
    domain: str = "general"
    requires_knowledge: bool = False
    requires_calculation: bool = False
    
    # 复杂度分数
    complexity_score: float = 0.0
    complexity_level: ComplexityLevel = ComplexityLevel.SIMPLE


@dataclass
class AdaptiveConfig:
    """自适应推理配置"""
    # 复杂度阈值
    simple_threshold: float = 0.3
    moderate_threshold: float = 0.5
    complex_threshold: float = 0.7
    
    # 推理参数
    direct_max_tokens: int = 30
    simple_cot_max_tokens: int = 60
    deep_cot_max_tokens: int = 100
    multi_step_max_tokens: int = 150
    
    # 性能优化
    enable_caching: bool = True
    cache_size: int = 100
    
    # 自适应学习
    enable_learning: bool = True
    learning_rate: float = 0.1


class ComplexityAnalyzer:
    """
    问题复杂度分析器
    
    分析问题的多个维度来评估复杂度：
    1. 文本复杂度 - 长度、句子结构
    2. 问题类型 - 是否需要推理、计算
    3. 知识需求 - 是否需要外部知识
    4. 领域识别 - 识别问题领域
    """
    
    def __init__(self):
        # 推理关键词
        self.reasoning_keywords = {
            '为什么', '怎么', '如何', '原因', '解释', '分析',
            '比较', '区别', '关系', '影响', '导致',
            'why', 'how', 'explain', 'analyze', 'compare',
            'reason', 'cause', 'effect', 'because', 'therefore'
        }
        
        # 计算关键词
        self.calculation_keywords = {
            '计算', '求', '多少', '加', '减', '乘', '除',
            '百分比', '比例', '平均', '总和',
            'calculate', 'compute', 'how many', 'how much',
            'add', 'subtract', 'multiply', 'divide', 'sum', 'average'
        }
        
        # 条件关键词
        self.conditional_keywords = {
            '如果', '假如', '假设', '若', '当', '条件',
            'if', 'when', 'suppose', 'assume', 'given', 'condition'
        }
        
        # 比较关键词
        self.comparison_keywords = {
            '比较', '对比', '区别', '不同', '相似', '更好',
            'compare', 'difference', 'similar', 'better', 'versus', 'vs'
        }
        
        # 领域关键词
        self.domain_keywords = {
            'math': {'数学', '计算', '方程', '函数', '几何', 'math', 'equation', 'function'},
            'physics': {'物理', '力', '能量', '速度', 'physics', 'force', 'energy', 'velocity'},
            'chemistry': {'化学', '分子', '原子', '反应', 'chemistry', 'molecule', 'atom'},
            'biology': {'生物', '细胞', '基因', '生物', 'biology', 'cell', 'gene', 'DNA'},
            'history': {'历史', '年代', '朝代', '战争', 'history', 'century', 'war'},
            'geography': {'地理', '国家', '城市', '气候', 'geography', 'country', 'climate'},
            'computer': {'计算机', '编程', '代码', '算法', 'computer', 'programming', 'code', 'algorithm'},
            'general': set()
        }
    
    def analyze(self, question: str) -> ComplexityFeatures:
        """
        分析问题复杂度
        
        Args:
            question: 问题文本
            
        Returns:
            复杂度特征
        """
        features = ComplexityFeatures()
        
        # 文本特征
        words = question.split()
        features.word_count = len(words)
        features.sentence_count = len(re.split(r'[.!?。！？]', question))
        features.avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # 问题特征
        question_lower = question.lower()
        
        features.has_multiple_questions = question.count('?') + question.count('？') > 1
        features.has_conditional = bool(self.conditional_keywords & set(question_lower.split()))
        features.has_comparison = bool(self.comparison_keywords & set(question_lower.split()))
        features.has_calculation = bool(self.calculation_keywords & set(question_lower.split()))
        features.has_reasoning_keywords = bool(self.reasoning_keywords & set(question_lower.split()))
        
        # 领域识别
        features.domain = self._identify_domain(question_lower)
        
        # 知识需求
        features.requires_knowledge = features.has_reasoning_keywords or features.domain != 'general'
        features.requires_calculation = features.has_calculation
        
        # 计算复杂度分数
        features.complexity_score = self._calculate_complexity_score(features)
        features.complexity_level = self._determine_complexity_level(features.complexity_score)
        
        return features
    
    def _identify_domain(self, text: str) -> str:
        """识别问题领域"""
        words = set(text.split())
        
        max_overlap = 0
        best_domain = 'general'
        
        for domain, keywords in self.domain_keywords.items():
            overlap = len(words & keywords)
            if overlap > max_overlap:
                max_overlap = overlap
                best_domain = domain
        
        return best_domain
    
    def _calculate_complexity_score(self, features: ComplexityFeatures) -> float:
        """
        计算复杂度分数 (0-1)
        
        考虑因素：
        - 文本长度
        - 问题类型
        - 知识需求
        - 领域复杂度
        """
        score = 0.0
        
        # 文本长度贡献 (0-0.2)
        length_score = min(features.word_count / 50, 1.0) * 0.2
        score += length_score
        
        # 多问题贡献 (0-0.15)
        if features.has_multiple_questions:
            score += 0.15
        
        # 条件推理贡献 (0-0.15)
        if features.has_conditional:
            score += 0.15
        
        # 比较贡献 (0-0.1)
        if features.has_comparison:
            score += 0.1
        
        # 计算贡献 (0-0.15)
        if features.has_calculation:
            score += 0.15
        
        # 推理关键词贡献 (0-0.15)
        if features.has_reasoning_keywords:
            score += 0.15
        
        # 知识需求贡献 (0-0.1)
        if features.requires_knowledge:
            score += 0.1
        
        return min(score, 1.0)
    
    def _determine_complexity_level(self, score: float) -> ComplexityLevel:
        """确定复杂度级别"""
        if score < 0.3:
            return ComplexityLevel.SIMPLE
        elif score < 0.5:
            return ComplexityLevel.MODERATE
        elif score < 0.7:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX


class AdaptiveReasoner:
    """
    自适应推理器
    
    根据问题复杂度自动选择最优推理方法：
    - 简单问题 → 直接回答（快速）
    - 中等问题 → 简单思维链
    - 复杂问题 → 深度思维链
    - 非常复杂 → 多步推理 + 工具辅助
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        memory_system=None,
        tool_manager=None,
        config: Optional[AdaptiveConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.memory_system = memory_system
        self.tool_manager = tool_manager
        self.config = config or AdaptiveConfig()
        
        # 复杂度分析器
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # 方法选择历史
        self._selection_history: List[Dict] = []
        
        # 性能统计
        self._performance_stats: Dict[ReasoningMethod, Dict] = {
            method: {'count': 0, 'total_time': 0, 'success': 0}
            for method in ReasoningMethod
        }
        
        # 缓存
        self._cache: Dict[str, Tuple[str, float]] = {}
    
    def select_method(self, features: ComplexityFeatures) -> ReasoningMethod:
        """
        选择推理方法
        
        Args:
            features: 复杂度特征
            
        Returns:
            推荐的推理方法
        """
        level = features.complexity_level
        
        # 基于复杂度级别选择方法
        if level == ComplexityLevel.SIMPLE:
            # 简单问题：直接回答
            if features.requires_calculation:
                return ReasoningMethod.SIMPLE_COT
            return ReasoningMethod.DIRECT
        
        elif level == ComplexityLevel.MODERATE:
            # 中等问题：简单思维链
            if features.requires_calculation:
                return ReasoningMethod.SIMPLE_COT
            if features.requires_knowledge:
                return ReasoningMethod.SIMPLE_COT
            return ReasoningMethod.DIRECT
        
        elif level == ComplexityLevel.COMPLEX:
            # 复杂问题：深度思维链
            if features.has_multiple_questions:
                return ReasoningMethod.MULTI_STEP
            if features.requires_knowledge and self.tool_manager:
                return ReasoningMethod.TOOL_ASSISTED
            return ReasoningMethod.DEEP_COT
        
        else:  # VERY_COMPLEX
            # 非常复杂：多步推理 + 工具
            return ReasoningMethod.MULTI_STEP
    
    async def reason(
        self,
        question: str,
        context: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        自适应推理
        
        自动分析问题复杂度并选择最优方法
        """
        start_time = time.time()
        
        # 分析复杂度
        features = self.complexity_analyzer.analyze(question)
        
        # 选择方法
        method = self.select_method(features)
        
        # 输出分析结果
        yield {
            "type": "analysis",
            "complexity_score": features.complexity_score,
            "complexity_level": features.complexity_level.value,
            "selected_method": method.value,
            "features": {
                "domain": features.domain,
                "requires_calculation": features.requires_calculation,
                "requires_knowledge": features.requires_knowledge
            }
        }
        
        # 执行推理
        result = ""
        async for chunk in self._execute_method(method, question, context, features):
            if chunk.get("type") == "content":
                result += chunk.get("content", "")
            yield chunk
        
        # 记录性能
        elapsed = time.time() - start_time
        self._record_performance(method, elapsed, True)
        
        # 记录选择历史
        self._selection_history.append({
            "question": question[:50],
            "complexity": features.complexity_score,
            "method": method.value,
            "elapsed": elapsed
        })
        
        yield {
            "type": "done",
            "method": method.value,
            "elapsed": elapsed
        }
    
    async def _execute_method(
        self,
        method: ReasoningMethod,
        question: str,
        context: Optional[str],
        features: ComplexityFeatures
    ) -> AsyncGenerator[Dict, None]:
        """执行选定的推理方法"""
        
        if method == ReasoningMethod.DIRECT:
            async for chunk in self._direct_reason(question, context):
                yield chunk
        
        elif method == ReasoningMethod.SIMPLE_COT:
            async for chunk in self._simple_cot_reason(question, context):
                yield chunk
        
        elif method == ReasoningMethod.DEEP_COT:
            async for chunk in self._deep_cot_reason(question, context, features):
                yield chunk
        
        elif method == ReasoningMethod.MULTI_STEP:
            async for chunk in self._multi_step_reason(question, context, features):
                yield chunk
        
        elif method == ReasoningMethod.TOOL_ASSISTED:
            async for chunk in self._tool_assisted_reason(question, context, features):
                yield chunk
    
    async def _direct_reason(
        self,
        question: str,
        context: Optional[str]
    ) -> AsyncGenerator[Dict, None]:
        """直接回答 - 最快"""
        prompt = question
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}"
        
        yield {"type": "method", "method": "direct", "description": "直接回答"}
        
        response = await self._generate(prompt, self.config.direct_max_tokens)
        
        yield {"type": "content", "content": response}
    
    async def _simple_cot_reason(
        self,
        question: str,
        context: Optional[str]
    ) -> AsyncGenerator[Dict, None]:
        """简单思维链 - 平衡速度和准确性"""
        prompt = f"""{question}

Let me think briefly:
"""
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        yield {"type": "method", "method": "simple_cot", "description": "简单思维链"}
        
        response = await self._generate(prompt, self.config.simple_cot_max_tokens)
        
        yield {"type": "content", "content": response}
    
    async def _deep_cot_reason(
        self,
        question: str,
        context: Optional[str],
        features: ComplexityFeatures
    ) -> AsyncGenerator[Dict, None]:
        """深度思维链 - 更详细的推理"""
        domain_hint = self._get_domain_hint(features.domain)
        
        prompt = f"""{question}

{domain_hint}

Let me think step by step:
1. First, I need to understand what is being asked.
2. Then, I recall relevant knowledge.
3. Next, I analyze the key points.
4. Finally, I provide the answer.

Step-by-step reasoning:
"""
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        yield {"type": "method", "method": "deep_cot", "description": "深度思维链"}
        
        response = await self._generate(prompt, self.config.deep_cot_max_tokens)
        
        yield {"type": "content", "content": response}
    
    async def _multi_step_reason(
        self,
        question: str,
        context: Optional[str],
        features: ComplexityFeatures
    ) -> AsyncGenerator[Dict, None]:
        """多步推理 - 最详细"""
        yield {"type": "method", "method": "multi_step", "description": "多步推理"}
        
        # 步骤1: 理解问题
        yield {"type": "step", "step": 1, "name": "理解问题"}
        understand_prompt = f"Question: {question}\n\nWhat is the core of this question?"
        understand_response = await self._generate(understand_prompt, 50)
        yield {"type": "step_content", "step": 1, "content": understand_response}
        
        # 步骤2: 分析
        yield {"type": "step", "step": 2, "name": "分析问题"}
        analyze_prompt = f"{question}\n\nKey points to consider:"
        analyze_response = await self._generate(analyze_prompt, 60)
        yield {"type": "step_content", "step": 2, "content": analyze_response}
        
        # 步骤3: 推理
        yield {"type": "step", "step": 3, "name": "推理"}
        reason_prompt = f"{question}\n\nBased on the analysis, my reasoning is:"
        reason_response = await self._generate(reason_prompt, 80)
        yield {"type": "step_content", "step": 3, "content": reason_response}
        
        # 步骤4: 结论
        yield {"type": "step", "step": 4, "name": "结论"}
        conclude_prompt = f"{question}\n\nFinal answer:"
        conclude_response = await self._generate(conclude_prompt, 50)
        yield {"type": "content", "content": conclude_response}
    
    async def _tool_assisted_reason(
        self,
        question: str,
        context: Optional[str],
        features: ComplexityFeatures
    ) -> AsyncGenerator[Dict, None]:
        """工具辅助推理"""
        yield {"type": "method", "method": "tool_assisted", "description": "工具辅助推理"}
        
        # 搜索知识
        knowledge = ""
        if self.tool_manager:
            yield {"type": "step", "step": 1, "name": "搜索知识"}
            try:
                wiki_result = await self.tool_manager.search_wikipedia(question)
                if wiki_result:
                    knowledge = wiki_result.get('summary', '')[:500]
                    yield {"type": "knowledge", "content": f"找到相关知识: {knowledge[:100]}..."}
            except Exception as e:
                yield {"type": "warning", "content": f"知识搜索失败: {str(e)}"}
        
        # 基于知识推理
        prompt = f"""{question}

Relevant knowledge: {knowledge}

Based on this knowledge, let me reason:
"""
        
        response = await self._generate(prompt, self.config.deep_cot_max_tokens)
        
        yield {"type": "content", "content": response}
    
    async def _generate(self, prompt: str, max_tokens: int) -> str:
        """生成文本"""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 只返回新生成的部分
        if response.startswith(prompt):
            return response[len(prompt):].strip()
        
        return response.strip()
    
    def _get_domain_hint(self, domain: str) -> str:
        """获取领域提示"""
        hints = {
            'math': 'This is a mathematics question. I need to apply mathematical principles.',
            'physics': 'This is a physics question. I need to consider physical laws.',
            'chemistry': 'This is a chemistry question. I need to think about chemical reactions.',
            'biology': 'This is a biology question. I need to consider biological processes.',
            'history': 'This is a history question. I need to recall historical facts.',
            'geography': 'This is a geography question. I need to consider geographical factors.',
            'computer': 'This is a computer science question. I need to apply computational thinking.',
            'general': ''
        }
        return hints.get(domain, '')
    
    def _record_performance(
        self,
        method: ReasoningMethod,
        elapsed: float,
        success: bool
    ) -> None:
        """记录性能"""
        stats = self._performance_stats[method]
        stats['count'] += 1
        stats['total_time'] += elapsed
        if success:
            stats['success'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {}
        
        for method, data in self._performance_stats.items():
            if data['count'] > 0:
                stats[method.value] = {
                    'count': data['count'],
                    'avg_time': data['total_time'] / data['count'],
                    'success_rate': data['success'] / data['count']
                }
        
        return {
            'method_stats': stats,
            'total_questions': sum(d['count'] for d in self._performance_stats.values()),
            'selection_history': self._selection_history[-10:]  # 最近10条
        }


class AdaptiveStrategyLearner:
    """
    自适应策略学习器
    
    从历史数据中学习最优策略选择
    """
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        
        # 方法-复杂度映射的Q值
        self.q_table: Dict[Tuple[ComplexityLevel, ReasoningMethod], float] = {}
        
        # 初始化Q值
        for level in ComplexityLevel:
            for method in ReasoningMethod:
                self.q_table[(level, method)] = 0.5  # 初始值
    
    def update(
        self,
        level: ComplexityLevel,
        method: ReasoningMethod,
        reward: float
    ) -> None:
        """
        更新Q值
        
        Args:
            level: 复杂度级别
            method: 使用的方法
            reward: 奖励（基于准确性和时间）
        """
        key = (level, method)
        current_q = self.q_table.get(key, 0.5)
        
        # Q-learning更新
        new_q = current_q + self.learning_rate * (reward - current_q)
        self.q_table[key] = new_q
    
    def get_best_method(self, level: ComplexityLevel) -> ReasoningMethod:
        """获取最佳方法"""
        best_method = ReasoningMethod.DIRECT
        best_q = -float('inf')
        
        for method in ReasoningMethod:
            q = self.q_table.get((level, method), 0.5)
            if q > best_q:
                best_q = q
                best_method = method
        
        return best_method
    
    def compute_reward(
        self,
        correct: bool,
        elapsed_time: float,
        expected_time: float
    ) -> float:
        """
        计算奖励
        
        Args:
            correct: 是否正确
            elapsed_time: 实际时间
            expected_time: 预期时间
            
        Returns:
            奖励值 (0-1)
        """
        # 准确性奖励
        accuracy_reward = 1.0 if correct else 0.0
        
        # 时间奖励（越快越好）
        time_ratio = expected_time / max(elapsed_time, 0.1)
        time_reward = min(time_ratio, 1.0)
        
        # 综合奖励
        reward = 0.7 * accuracy_reward + 0.3 * time_reward
        
        return reward
