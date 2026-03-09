#!/usr/bin/env python3
"""
自适应推理测试脚本
测试根据问题复杂度自动选择方法的效果
"""

import sys
import os
import time
import asyncio
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 测试问题集（按复杂度分类）
TEST_QUESTIONS = {
    "simple": [
        ("What is 2 + 2?", "4"),
        ("What color is the sky?", "BLUE"),
        ("What is the capital of France?", "PARIS"),
        ("How many days in a week?", "7"),
        ("What is the opposite of hot?", "COLD"),
    ],
    "moderate": [
        ("What is 15 + 27?", "42"),
        ("What is the SI unit of force?", "NEWTON"),
        ("What is the powerhouse of the cell?", "MITOCHONDRIA"),
        ("What does CPU stand for?", "CENTRAL PROCESSING UNIT"),
        ("How many planets are in our solar system?", "8"),
    ],
    "complex": [
        ("Explain why the sky appears blue during the day.", "SCATTERING"),
        ("Compare and contrast DNA and RNA.", "GENETIC"),
        ("What causes the seasons on Earth?", "TILT"),
        ("How does photosynthesis work?", "CHLOROPHYLL"),
        ("Explain the water cycle.", "EVAPORATION"),
    ],
    "very_complex": [
        ("If a train travels at 60 mph for 2.5 hours, how far does it go, and how long would it take to travel 180 miles?", "150"),
        ("Analyze the relationship between economic growth and environmental sustainability.", "BALANCE"),
        ("Compare the advantages and disadvantages of renewable vs non-renewable energy sources.", "RENEWABLE"),
        ("Explain how the human immune system responds to a viral infection.", "IMMUNE"),
        ("What factors contributed to the fall of the Roman Empire?", "MULTIPLE"),
    ]
}


async def test_adaptive_reasoning():
    """测试自适应推理"""
    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    model_path = './models/qwen'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    print("Model loaded!\n")
    
    # 导入自适应推理器
    from core.adaptive_reasoner import AdaptiveReasoner, AdaptiveConfig, ComplexityLevel
    
    config = AdaptiveConfig()
    reasoner = AdaptiveReasoner(model, tokenizer, config=config)
    
    print("="*60)
    print("  Adaptive Reasoning Test")
    print("="*60)
    
    results = {
        "simple": {"correct": 0, "total": 0, "time": 0},
        "moderate": {"correct": 0, "total": 0, "time": 0},
        "complex": {"correct": 0, "total": 0, "time": 0},
        "very_complex": {"correct": 0, "total": 0, "time": 0}
    }
    
    method_usage = {}
    
    for complexity, questions in TEST_QUESTIONS.items():
        print(f"\n[{complexity.upper()} QUESTIONS]")
        print("-"*40)
        
        for question, expected_answer in questions:
            start_time = time.time()
            
            # 使用自适应推理
            result_text = ""
            selected_method = ""
            complexity_score = 0
            
            async for chunk in reasoner.reason(question):
                if chunk.get("type") == "analysis":
                    complexity_score = chunk.get("complexity_score", 0)
                    selected_method = chunk.get("selected_method", "unknown")
                    print(f"  Complexity: {complexity_score:.2f} → Method: {selected_method}")
                elif chunk.get("type") == "content":
                    result_text = chunk.get("content", "")
                elif chunk.get("type") == "done":
                    pass
            
            elapsed = time.time() - start_time
            
            # 检查答案
            is_correct = expected_answer.upper() in result_text.upper()
            
            results[complexity]["total"] += 1
            results[complexity]["time"] += elapsed
            if is_correct:
                results[complexity]["correct"] += 1
            
            # 记录方法使用
            if selected_method not in method_usage:
                method_usage[selected_method] = 0
            method_usage[selected_method] += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} Q: {question[:40]}...")
            print(f"     Expected: {expected_answer}, Got: {result_text[:50]}...")
            print(f"     Time: {elapsed:.2f}s")
            print()
    
    # 打印结果汇总
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    
    total_correct = 0
    total_questions = 0
    total_time = 0
    
    for complexity, data in results.items():
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        avg_time = data["time"] / data["total"] if data["total"] > 0 else 0
        print(f"  {complexity:15s}: {acc:.0%} ({data['correct']}/{data['total']}), avg {avg_time:.2f}s")
        
        total_correct += data["correct"]
        total_questions += data["total"]
        total_time += data["time"]
    
    print("-"*60)
    overall_acc = total_correct / total_questions if total_questions > 0 else 0
    print(f"  {'OVERALL':15s}: {overall_acc:.0%} ({total_correct}/{total_questions})")
    
    print("\n  Method Usage:")
    for method, count in sorted(method_usage.items(), key=lambda x: -x[1]):
        print(f"    {method}: {count}")
    
    print("="*60)
    
    # 获取详细统计
    stats = reasoner.get_stats()
    print("\n  Detailed Stats:")
    for method, data in stats.get('method_stats', {}).items():
        print(f"    {method}: count={data['count']}, avg_time={data['avg_time']:.2f}s")
    
    return results


def main():
    """主函数"""
    asyncio.run(test_adaptive_reasoning())


if __name__ == '__main__':
    main()
