#!/usr/bin/env python3
"""
流式推理优化基准测试
利用逐步推理优势提升性能
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


class StreamingBenchmarkRunner:
    """流式推理基准测试运行器"""
    
    def __init__(self, model_path: str = "./models/qwen"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.results = {}
        
    def load_model(self):
        """加载模型"""
        print("Loading model for streaming benchmark...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32
        )
        self.model.eval()
        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B parameters")
    
    def stream_generate_with_reasoning(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7
    ) -> str:
        """
        流式生成并逐步推理
        
        使用思维链提示增强推理能力
        """
        # 构建思维链提示
        cot_prompt = f"""{prompt}

让我一步步思考：

1. 首先，我需要理解问题的关键点...
2. 然后，分析相关信息...
3. 接着，进行推理...
4. 最后，得出结论...

思考过程："""
        
        inputs = self.tokenizer(cot_prompt, return_tensors='pt')
        
        with self.torch.no_grad():
            # 第一阶段：生成思考过程
            thought_outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            thought_text = self.tokenizer.decode(thought_outputs[0], skip_special_tokens=True)
        
        # 第二阶段：基于思考生成答案
        answer_prompt = f"{thought_text}\n\n基于以上思考，答案是："
        answer_inputs = self.tokenizer(answer_prompt, return_tensors='pt')
        
        with self.torch.no_grad():
            answer_outputs = self.model.generate(
                **answer_inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            answer_text = self.tokenizer.decode(answer_outputs[0], skip_special_tokens=True)
        
        return answer_text
    
    def extract_answer_with_reasoning(
        self,
        response: str,
        choices: list = None,
        question_type: str = "multiple_choice"
    ) -> str:
        """
        从推理响应中提取答案
        
        支持多种问题类型
        """
        response = response.upper()
        
        if question_type == "multiple_choice":
            # 多选题：提取A, B, C, D
            for letter in ['A', 'B', 'C', 'D']:
                if letter in response:
                    return letter
            return 'A'
        
        elif question_type == "number":
            # 数字题：提取数字
            import re
            numbers = re.findall(r'\d+', response)
            return numbers[-1] if numbers else "0"
        
        elif question_type == "true_false":
            # 判断题
            if '正确' in response or 'TRUE' in response or '对' in response:
                return "正确"
            elif '错误' in response or 'FALSE' in response or '错' in response:
                return "错误"
            return "正确"
        
        return response[:50]
    
    def run_mmlu_streaming(self) -> dict:
        """运行MMLU测试（流式推理优化）"""
        print("\n" + "="*50)
        print("Running MMLU Benchmark (Streaming Reasoning)")
        print("="*50)
        
        correct = 0
        total = len(MMLU_DATA)
        category_scores = {}
        
        for item in MMLU_DATA:
            category = item.get('category', 'general')
            if category not in category_scores:
                category_scores[category] = {'correct': 0, 'total': 0}
            
            # 使用思维链提示
            prompt = f"""问题：{item['question']}
选项：{', '.join(item['choices'])}

请仔细分析每个选项，然后选择最正确的答案。"""
            
            response = self.stream_generate_with_reasoning(prompt)
            predicted = self.extract_answer_with_reasoning(response, item['choices'])
            is_correct = predicted == item['answer']
            
            if is_correct:
                correct += 1
                category_scores[category]['correct'] += 1
            category_scores[category]['total'] += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} {item['id']}: Predicted={predicted}, Correct={item['answer']}")
        
        accuracy = correct / total
        
        print(f"\nMMLU Results (Streaming):")
        print(f"  Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
        for cat, scores in category_scores.items():
            cat_acc = scores['correct'] / scores['total'] if scores['total'] > 0 else 0
            print(f"  {cat}: {cat_acc:.2%} ({scores['correct']}/{scores['total']})")
        
        return {
            'benchmark': 'MMLU_Streaming',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'category_scores': category_scores
        }
    
    def run_gsm8k_streaming(self) -> dict:
        """运行GSM8K测试（流式推理优化）"""
        print("\n" + "="*50)
        print("Running GSM8K Benchmark (Streaming Reasoning)")
        print("="*50)
        
        correct = 0
        total = len(GSM8K_DATA)
        
        for item in GSM8K_DATA:
            # 数学推理使用详细的思维链
            prompt = f"""数学问题：{item['question']}

请一步步计算，写出详细的计算过程。"""
            
            response = self.stream_generate_with_reasoning(prompt, max_new_tokens=200)
            predicted = self.extract_answer_with_reasoning(response, question_type="number")
            is_correct = predicted == item['answer']
            
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} {item['id']}: Predicted={predicted}, Correct={item['answer']}")
        
        accuracy = correct / total
        print(f"\nGSM8K Results (Streaming): {accuracy:.2%} ({correct}/{total})")
        
        return {
            'benchmark': 'GSM8K_Streaming',
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def run_arc_streaming(self) -> dict:
        """运行ARC测试（流式推理优化）"""
        print("\n" + "="*50)
        print("Running ARC Benchmark (Streaming Reasoning)")
        print("="*50)
        
        correct = 0
        total = len(ARC_DATA)
        
        for item in ARC_DATA:
            prompt = f"""科学问题：{item['question']}
选项：{', '.join(item['choices'])}

请运用科学知识分析每个选项的正确性。"""
            
            response = self.stream_generate_with_reasoning(prompt)
            predicted = self.extract_answer_with_reasoning(response, item['choices'])
            is_correct = predicted == item['answer']
            
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} {item['id']}: Predicted={predicted}, Correct={item['answer']}")
        
        accuracy = correct / total
        print(f"\nARC Results (Streaming): {accuracy:.2%} ({correct}/{total})")
        
        return {
            'benchmark': 'ARC_Streaming',
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def run_truthfulqa_streaming(self) -> dict:
        """运行TruthfulQA测试（流式推理优化）"""
        print("\n" + "="*50)
        print("Running TruthfulQA Benchmark (Streaming Reasoning)")
        print("="*50)
        
        correct = 0
        total = len(TRUTHFULQA_DATA)
        
        for item in TRUTHFULQA_DATA:
            prompt = f"""事实问题：{item['question']}
选项：{', '.join(item['choices'])}

请根据科学事实选择正确答案，避免常见误解。"""
            
            response = self.stream_generate_with_reasoning(prompt)
            predicted = self.extract_answer_with_reasoning(response, item['choices'])
            is_correct = predicted == item['answer']
            
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} {item['id']}: Predicted={predicted}, Correct={item['answer']}")
        
        accuracy = correct / total
        print(f"\nTruthfulQA Results (Streaming): {accuracy:.2%} ({correct}/{total})")
        
        return {
            'benchmark': 'TruthfulQA_Streaming',
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def run_all_benchmarks_streaming(self) -> dict:
        """运行所有基准测试（流式推理优化）"""
        print("\n" + "="*60)
        print("  DigBrain Streaming Reasoning Benchmark Suite")
        print("="*60)
        
        start_time = time.time()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_path,
            'method': 'streaming_reasoning',
            'benchmarks': {}
        }
        
        # 运行各项测试
        results['benchmarks']['MMLU_Streaming'] = self.run_mmlu_streaming()
        results['benchmarks']['GSM8K_Streaming'] = self.run_gsm8k_streaming()
        results['benchmarks']['ARC_Streaming'] = self.run_arc_streaming()
        results['benchmarks']['TruthfulQA_Streaming'] = self.run_truthfulqa_streaming()
        
        total_time = time.time() - start_time
        
        # 计算平均分数
        avg_accuracy = sum(
            r['accuracy'] for r in results['benchmarks'].values()
        ) / len(results['benchmarks'])
        
        results['total_time'] = total_time
        results['average_accuracy'] = avg_accuracy
        
        # 打印汇总
        print("\n" + "="*60)
        print("  STREAMING BENCHMARK SUMMARY")
        print("="*60)
        
        for name, result in results['benchmarks'].items():
            print(f"  {name:25s}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
        
        print("-"*60)
        print(f"  {'Average':25s}: {avg_accuracy:.2%}")
        print(f"  {'Total Time':25s}: {total_time:.1f}s")
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
    
    parser = argparse.ArgumentParser(description='Run Streaming Reasoning Benchmarks')
    parser.add_argument('--model', type=str, default='./models/qwen',
                        help='Model path')
    parser.add_argument('--output', type=str, default='./results/benchmarks/streaming_results.json',
                        help='Output file path')
    
    args = parser.parse_args()
    
    # 运行测试
    runner = StreamingBenchmarkRunner(args.model)
    runner.load_model()
    results = runner.run_all_benchmarks_streaming()
    runner.save_results(results, args.output)
    
    return results


if __name__ == '__main__':
    main()
