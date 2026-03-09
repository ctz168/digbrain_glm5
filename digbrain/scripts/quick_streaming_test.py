#!/usr/bin/env python3
"""
轻量级流式推理测试
快速验证流式推理优化效果
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quick_streaming_test():
    """快速流式推理测试"""
    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    model_path = './models/qwen'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    
    print("Model loaded!\n")
    print("="*50)
    print("Streaming Reasoning Optimization Test")
    print("="*50)
    
    # 测试问题
    tests = [
        {
            "question": "What is 15 + 27?",
            "answer": "42",
            "type": "math"
        },
        {
            "question": "What is the SI unit of force?",
            "answer": "NEWTON",
            "type": "physics"
        },
        {
            "question": "What is the powerhouse of the cell?",
            "answer": "MITOCHONDRIA",
            "type": "biology"
        }
    ]
    
    # 方法1: 直接提问
    print("\n[Method 1: Direct Question]")
    direct_correct = 0
    direct_times = []
    
    for test in tests:
        start = time.time()
        inputs = tokenizer(test["question"], return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start
        direct_times.append(elapsed)
        
        is_correct = test["answer"] in response.upper()
        if is_correct:
            direct_correct += 1
        print(f"  {'✓' if is_correct else '✗'} {test['type']}: {response[-50:]}")
    
    direct_acc = direct_correct / len(tests)
    direct_avg_time = sum(direct_times) / len(direct_times)
    
    # 方法2: 思维链推理
    print("\n[Method 2: Chain-of-Thought Reasoning]")
    cot_correct = 0
    cot_times = []
    
    for test in tests:
        # 思维链提示
        cot_prompt = f"""Question: {test["question"]}

Let me think step by step:
1. First, I need to understand what is being asked.
2. Then, I recall relevant knowledge.
3. Finally, I provide the answer.

Step-by-step reasoning:"""
        
        start = time.time()
        inputs = tokenizer(cot_prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=60, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start
        cot_times.append(elapsed)
        
        is_correct = test["answer"] in response.upper()
        if is_correct:
            cot_correct += 1
        print(f"  {'✓' if is_correct else '✗'} {test['type']}: {response[-80:]}")
    
    cot_acc = cot_correct / len(tests)
    cot_avg_time = sum(cot_times) / len(cot_times)
    
    # 结果对比
    print("\n" + "="*50)
    print("Results Comparison")
    print("="*50)
    print(f"Direct Question:")
    print(f"  Accuracy: {direct_acc:.2%}")
    print(f"  Avg Time: {direct_avg_time:.2f}s")
    print(f"\nChain-of-Thought:")
    print(f"  Accuracy: {cot_acc:.2%}")
    print(f"  Avg Time: {cot_avg_time:.2f}s")
    
    improvement = (cot_acc - direct_acc) / max(direct_acc, 0.01) * 100
    print(f"\nImprovement: {improvement:+.1f}%")
    
    return {
        "direct_accuracy": direct_acc,
        "cot_accuracy": cot_acc,
        "improvement": improvement
    }


if __name__ == '__main__':
    quick_streaming_test()
