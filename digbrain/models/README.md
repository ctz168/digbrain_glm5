# Qwen2.5-0.5B-Instruct 模型权重

由于模型文件较大（约1GB），请使用以下方法下载：

## 方法1: 使用HuggingFace CLI
```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/qwen
```

## 方法2: 使用Python脚本
```bash
python scripts/download_model.py --model "Qwen/Qwen2.5-0.5B-Instruct" --save-path "./models/qwen"
```

## 方法3: 直接下载
从 HuggingFace Hub 下载: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

## 模型信息
- 参数量: 0.49B
- 支持语言: 中文、英文
- 支持任务: 文本生成、对话、推理
- 许可证: Apache 2.0
