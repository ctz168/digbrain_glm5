#!/usr/bin/env python3
"""
完整基准测试运行器
运行所有标准基准测试并生成报告
"""

import sys
import os
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入测试数据
from data.benchmarks.test_data import (
    MMLU_DATA, HELLASWAG_DATA, WINOGRANDE_DATA, 
    ARC_DATA, GSM8K_DATA, TRUTHFULQA_DATA
)

class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self, model_path: str = "./models/qwen"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.results = {}
        
    def load_model(self):
        """加载模型"""
        print("Loading model...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # 保存torch引用供后续使用
        self.torch = torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32
        )
        self.model.eval()
        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B parameters")
    
    def generate_answer(self, prompt: str, max_new_tokens: int = 50) -> str:
        """生成答案"""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def extract_choice(self, response: str, choices: list) -> str:
        """从响应中提取选项"""
        response = response.upper()
        
        # 尝试匹配A, B, C, D
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response:
                return letter
        
        # 尝试匹配选项内容
        for i, choice in enumerate(choices):
            choice_text = choice.split('. ', 1)[-1] if '. ' in choice else choice
            if choice_text.lower() in response.lower():
                return chr(65 + i)  # A, B, C, D
        
        return 'A'  # 默认返回A
    
    def run_mmlu(self) -> dict:
        """运行MMLU测试"""
        print("\n" + "="*50)
        print("Running MMLU Benchmark")
        print("="*50)
        
        correct = 0
        total = len(MMLU_DATA)
        category_scores = {}
        
        for item in MMLU_DATA:
            category = item.get('category', 'general')
            if category not in category_scores:
                category_scores[category] = {'correct': 0, 'total': 0}
            
            prompt = f"""Question: {item['question']}
Choices: {', '.join(item['choices'])}
Answer with just the letter (A, B, C, or D).
Answer:"""
            
            response = self.generate_answer(prompt, max_new_tokens=10)
            predicted = self.extract_choice(response, item['choices'])
            is_correct = predicted == item['answer']
            
            if is_correct:
                correct += 1
                category_scores[category]['correct'] += 1
            category_scores[category]['total'] += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} {item['id']}: Predicted={predicted}, Correct={item['answer']}")
        
        accuracy = correct / total
        
        print(f"\nMMLU Results:")
        print(f"  Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
        for cat, scores in category_scores.items():
            cat_acc = scores['correct'] / scores['total'] if scores['total'] > 0 else 0
            print(f"  {cat}: {cat_acc:.2%} ({scores['correct']}/{scores['total']})")
        
        return {
            'benchmark': 'MMLU',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'category_scores': category_scores
        }
    
    def run_hellaswag(self) -> dict:
        """运行HellaSwag测试"""
        print("\n" + "="*50)
        print("Running HellaSwag Benchmark")
        print("="*50)
        
        correct = 0
        total = len(HELLASWAG_DATA)
        
        for item in HELLASWAG_DATA:
            prompt = f"""Complete the following sentence with the most logical ending:

"{item['context']}"

Possible endings:
A. {item['endings'][0]}
B. {item['endings'][1]}
C. {item['endings'][2]}
D. {item['endings'][3]}

Choose the most logical ending (A, B, C, or D):"""
            
            response = self.generate_answer(prompt, max_new_tokens=10)
            predicted_idx = ord(self.extract_choice(response, item['endings'])) - 65
            is_correct = predicted_idx == item['label']
            
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} {item['id']}: Predicted={predicted_idx}, Correct={item['label']}")
        
        accuracy = correct / total
        print(f"\nHellaSwag Results: {accuracy:.2%} ({correct}/{total})")
        
        return {
            'benchmark': 'HellaSwag',
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def run_winogrande(self) -> dict:
        """运行WinoGrande测试"""
        print("\n" + "="*50)
        print("Running WinoGrande Benchmark")
        print("="*50)
        
        correct = 0
        total = len(WINOGRANDE_DATA)
        
        for item in WINOGRANDE_DATA:
            prompt = f"""Fill in the blank with the correct option:

"{item['sentence'].replace('___', '[BLANK]')}"

Options:
1. {item['option1']}
2. {item['option2']}

Which option should fill the blank? Answer with 1 or 2:"""
            
            response = self.generate_answer(prompt, max_new_tokens=10)
            
            # 提取答案
            if '1' in response:
                predicted = 1
            elif '2' in response:
                predicted = 2
            else:
                predicted = 1  # 默认
            
            is_correct = predicted == item['answer']
            
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} {item['id']}: Predicted={predicted}, Correct={item['answer']}")
        
        accuracy = correct / total
        print(f"\nWinoGrande Results: {accuracy:.2%} ({correct}/{total})")
        
        return {
            'benchmark': 'WinoGrande',
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def run_arc(self) -> dict:
        """运行ARC测试"""
        print("\n" + "="*50)
        print("Running ARC Benchmark")
        print("="*50)
        
        correct = 0
        total = len(ARC_DATA)
        
        for item in ARC_DATA:
            prompt = f"""Question: {item['question']}
Choices: {', '.join(item['choices'])}
Answer with just the letter (A, B, C, or D).
Answer:"""
            
            response = self.generate_answer(prompt, max_new_tokens=10)
            predicted = self.extract_choice(response, item['choices'])
            is_correct = predicted == item['answer']
            
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} {item['id']}: Predicted={predicted}, Correct={item['answer']}")
        
        accuracy = correct / total
        print(f"\nARC Results: {accuracy:.2%} ({correct}/{total})")
        
        return {
            'benchmark': 'ARC',
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def run_gsm8k(self) -> dict:
        """运行GSM8K测试"""
        print("\n" + "="*50)
        print("Running GSM8K Benchmark")
        print("="*50)
        
        correct = 0
        total = len(GSM8K_DATA)
        
        for item in GSM8K_DATA:
            prompt = f"""Solve this math problem step by step:

{item['question']}

Show your work and give the final answer as a number."""
            
            response = self.generate_answer(prompt, max_new_tokens=100)
            
            # 尝试从响应中提取数字
            import re
            numbers = re.findall(r'\d+', response)
            predicted = numbers[-1] if numbers else "0"
            
            is_correct = predicted == item['answer']
            
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} {item['id']}: Predicted={predicted}, Correct={item['answer']}")
        
        accuracy = correct / total
        print(f"\nGSM8K Results: {accuracy:.2%} ({correct}/{total})")
        
        return {
            'benchmark': 'GSM8K',
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def run_truthfulqa(self) -> dict:
        """运行TruthfulQA测试"""
        print("\n" + "="*50)
        print("Running TruthfulQA Benchmark")
        print("="*50)
        
        correct = 0
        total = len(TRUTHFULQA_DATA)
        
        for item in TRUTHFULQA_DATA:
            prompt = f"""Question: {item['question']}
Choices: {', '.join(item['choices'])}
Answer with just the letter (A, B, C, or D).
Answer:"""
            
            response = self.generate_answer(prompt, max_new_tokens=10)
            predicted = self.extract_choice(response, item['choices'])
            is_correct = predicted == item['answer']
            
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} {item['id']}: Predicted={predicted}, Correct={item['answer']}")
        
        accuracy = correct / total
        print(f"\nTruthfulQA Results: {accuracy:.2%} ({correct}/{total})")
        
        return {
            'benchmark': 'TruthfulQA',
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def run_all_benchmarks(self) -> dict:
        """运行所有基准测试"""
        print("\n" + "="*60)
        print("  DigBrain Complete Benchmark Suite")
        print("="*60)
        
        start_time = time.time()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_path,
            'benchmarks': {}
        }
        
        # 运行各项测试
        results['benchmarks']['MMLU'] = self.run_mmlu()
        results['benchmarks']['HellaSwag'] = self.run_hellaswag()
        results['benchmarks']['WinoGrande'] = self.run_winogrande()
        results['benchmarks']['ARC'] = self.run_arc()
        results['benchmarks']['GSM8K'] = self.run_gsm8k()
        results['benchmarks']['TruthfulQA'] = self.run_truthfulqa()
        
        total_time = time.time() - start_time
        
        # 计算平均分数
        avg_accuracy = sum(
            r['accuracy'] for r in results['benchmarks'].values()
        ) / len(results['benchmarks'])
        
        results['total_time'] = total_time
        results['average_accuracy'] = avg_accuracy
        
        # 打印汇总
        print("\n" + "="*60)
        print("  BENCHMARK SUMMARY")
        print("="*60)
        
        for name, result in results['benchmarks'].items():
            print(f"  {name:15s}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
        
        print("-"*60)
        print(f"  {'Average':15s}: {avg_accuracy:.2%}")
        print(f"  {'Total Time':15s}: {total_time:.1f}s")
        print("="*60)
        
        return results
    
    def save_results(self, results: dict, output_path: str):
        """保存结果"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run DigBrain Benchmarks')
    parser.add_argument('--model', type=str, default='./models/qwen',
                        help='Model path')
    parser.add_argument('--output', type=str, default='./results/benchmarks/results.json',
                        help='Output file path')
    
    args = parser.parse_args()
    
    # 运行测试
    runner = BenchmarkRunner(args.model)
    runner.load_model()
    results = runner.run_all_benchmarks()
    runner.save_results(results, args.output)
    
    return results


if __name__ == '__main__':
    main()
