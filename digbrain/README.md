# DigBrain - 类脑智能系统

<div align="center">

![DigBrain Logo](docs/images/logo.png)

**一个受人类大脑启发的开源人工智能框架**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[English](#english-documentation) | [中文文档](#中文文档)

</div>

---

## 中文文档

### 项目简介

DigBrain是一个创新的类脑智能系统，模拟人脑的信息处理机制，实现高刷新率流式处理、存算分离架构、在线STDP学习等核心特性。系统整合了Qwen3.5-0.8B作为核心推理模型，支持多模态输入（文本、图像、视频），并具备类人脑的记忆搜索和管理能力。

### 设计思想

#### 1. 高刷新率流式处理

人脑以毫秒级速度持续处理感官信息，DigBrain模拟这一机制，实现高刷新率的流式处理：

- **流式输入处理**：每次处理小批量数据，模拟人脑的实时感知
- **流式输出生成**：边推理边输出，提供即时响应
- **并行记忆检索**：在推理过程中并行搜索记忆库和外部知识源
- **刷新率可配置**：支持10Hz-100Hz的处理频率调节

```
输入流 → [感知层] → [记忆检索] → [推理层] → [输出流]
           ↑              ↑           ↑
        高刷新处理    并行搜索    STDP学习
```

#### 2. 存算分离架构

借鉴DeepSeek最新论文框架，实现存储与计算的分离：

- **记忆存储层**：独立的海马体式记忆存储系统
- **计算处理层**：轻量级推理引擎，按需加载记忆
- **动态索引**：基于语义的快速记忆检索
- **分布式扩展**：支持多节点记忆存储和检索

#### 3. 在线STDP学习

脉冲时序依赖可塑性（Spike-Timing-Dependent Plasticity）是在线学习的核心：

- **实时权重更新**：根据输入输出时序关系动态调整权重
- **赫布学习规则**："一起激发，一起连接"的神经可塑性
- **长期增强/抑制**：LTP/LTD机制实现记忆强化和遗忘
- **元学习支持**：学会如何学习，适应不同任务

#### 4. Qwen3.5-0.8B核心模型

整合Qwen3.5-0.8B作为核心语言和推理引擎：

- **原生多模态支持**：文本、图像统一处理
- **视频流处理**：逐帧解构，流式分析
- **高效推理**：0.8B参数量，适合边缘部署
- **可微调架构**：支持LoRA等高效微调方法

#### 5. 类人脑记忆系统

模拟海马体的记忆管理机制：

- **神经累积增长**：记忆以神经元连接形式累积存储
- **层级记忆结构**：短期记忆→长期记忆→永久记忆
- **联想检索**：基于语义相似性和时序关联的记忆搜索
- **遗忘机制**：模拟人脑的遗忘曲线，优化存储空间

#### 6. 无限知识扩展

通过维基百科API实现知识库的无限扩展：

- **实时知识检索**：按需获取最新信息
- **多语言支持**：支持中英文维基百科
- **知识融合**：将检索结果与内部记忆整合

### 项目亮点

#### ✅ 真实基准测试

我们承诺提供真实、可复现的基准测试结果：

| 测试基准 | DigBrain得分 | 基线模型 | 说明 |
|---------|-------------|---------|------|
| MMLU | 实测分数 | 对比基线 | 多任务语言理解 |
| HellaSwag | 实测分数 | 对比基线 | 常识推理 |
| WinoGrande | 实测分数 | 对比基线 | 核心指代消解 |
| ARC-Challenge | 实测分数 | 对比基线 | AI推理挑战 |
| TruthfulQA | 实测分数 | 对比基线 | 事实准确性 |
| GSM8K | 实测分数 | 对比基线 | 数学推理 |

**反作弊声明**：
- 所有测试使用官方评估脚本
- 测试代码完全开源
- 提供完整的测试日志和配置
- 支持第三方复现验证

#### 🚀 技术创新

1. **存算分离**：突破传统Transformer的内存瓶颈
2. **在线学习**：无需离线重训练，实时适应新知识
3. **高刷新处理**：模拟人脑的实时响应能力
4. **多模态融合**：统一处理文本、图像、视频

### 安装部署（Mac平台）

#### 系统要求

- macOS 12.0+
- Python 3.10+
- 8GB+ RAM（推荐16GB）
- 20GB+ 可用磁盘空间

#### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/ctz168/digbrain.git
cd digbrain

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载Qwen3.5-0.8B模型
python scripts/download_model.py

# 5. 初始化记忆存储
python scripts/init_memory.py

# 6. 运行测试
pytest tests/

# 7. 启动服务
python -m digbrain.server
```

#### Docker部署

```bash
# 构建镜像
docker build -t digbrain:latest .

# 运行容器
docker run -d -p 8000:8000 -v ./data:/app/data digbrain:latest
```

### 训练方法

#### 在线学习

```python
from digbrain import DigBrain

# 初始化系统
brain = DigBrain()

# 在线学习模式
brain.enable_online_learning(stdp_rate=0.01)

# 流式输入处理
for chunk in input_stream:
    response = brain.process(chunk)
    # STDP自动更新权重
```

#### 离线训练

```bash
# 单模块训练
python -m digbrain.training.train_memory --data ./data/memory_train.json
python -m digbrain.training.train_stdp --episodes 1000

# 综合多线程训练
python -m digbrain.training.train_all --config ./config/training.yaml
```

### API调用

#### RESTful API

```python
import requests

# 文本处理
response = requests.post(
    "http://localhost:8000/api/process",
    json={
        "input": "你好，请介绍一下自己",
        "stream": True,
        "search_wiki": True
    }
)

# 流式响应
for chunk in response.iter_content(chunk_size=None):
    print(chunk.decode(), end="")
```

#### Python SDK

```python
from digbrain import DigBrainClient

client = DigBrainClient("http://localhost:8000")

# 同步调用
result = client.process("什么是人工智能？")

# 流式调用
for chunk in client.stream_process("讲一个故事"):
    print(chunk, end="")

# 多模态输入
result = client.process_multimodal(
    text="描述这张图片",
    image="./image.jpg"
)
```

### 前端Web使用

启动Web界面：

```bash
python -m digbrain.web.server
```

访问 http://localhost:3000 即可使用图形界面。

功能特性：
- 💬 实时对话界面
- 🖼️ 图片上传分析
- 🎥 视频流处理
- 📊 记忆可视化
- ⚙️ 参数调节面板

### 项目结构

```
digbrain/
├── core/               # 核心处理模块
│   ├── brain.py       # 主脑控制器
│   ├── stream.py      # 流式处理引擎
│   └── attention.py   # 注意力机制
├── memory/            # 记忆系统
│   ├── hippocampus.py # 海马体模拟
│   ├── storage.py     # 存储后端
│   └── retrieval.py   # 记忆检索
├── training/          # 训练模块
│   ├── online.py      # 在线学习
│   ├── offline.py     # 离线训练
│   └── stdp.py        # STDP学习
├── evaluation/        # 评估模块
│   ├── benchmarks.py  # 基准测试
│   └── metrics.py     # 评估指标
├── api/               # API接口
│   ├── rest.py        # REST API
│   └── websocket.py   # WebSocket
├── web/               # Web前端
│   ├── static/        # 静态资源
│   └── templates/     # 模板文件
├── tools/             # 工具模块
│   ├── wiki_search.py # 维基百科搜索
│   └── web_tools.py   # 网页工具
├── models/            # 模型文件
│   └── qwen/          # Qwen模型
├── config/            # 配置文件
├── scripts/           # 脚本工具
├── tests/             # 测试文件
└── docs/              # 文档
```

### 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

### 致谢

- Qwen Team - 提供优秀的Qwen3.5-0.8B模型
- DeepSeek Team - 存算分离架构灵感
- 神经科学研究社区 - 类脑计算理论基础

---

## English Documentation

### Project Overview

DigBrain is an innovative brain-inspired intelligent system that simulates the human brain's information processing mechanisms. It implements core features such as high-refresh-rate streaming processing, storage-computation separation architecture, and online STDP learning. The system integrates Qwen3.5-0.8B as the core reasoning model, supports multimodal inputs (text, image, video), and possesses human-brain-like memory search and management capabilities.

### Design Philosophy

#### 1. High-Refresh-Rate Streaming Processing

The human brain processes sensory information at millisecond speeds continuously. DigBrain simulates this mechanism with high-refresh-rate streaming processing:

- **Streaming Input Processing**: Process small batches of data each time, simulating real-time perception
- **Streaming Output Generation**: Generate output while reasoning, providing immediate response
- **Parallel Memory Retrieval**: Search memory and external knowledge sources in parallel during reasoning
- **Configurable Refresh Rate**: Support 10Hz-100Hz processing frequency adjustment

#### 2. Storage-Computation Separation Architecture

Inspired by DeepSeek's latest paper framework, implementing separation of storage and computation:

- **Memory Storage Layer**: Independent hippocampus-like memory storage system
- **Computation Processing Layer**: Lightweight inference engine, loading memory on demand
- **Dynamic Indexing**: Fast memory retrieval based on semantics
- **Distributed Scaling**: Support multi-node memory storage and retrieval

#### 3. Online STDP Learning

Spike-Timing-Dependent Plasticity is the core of online learning:

- **Real-time Weight Update**: Dynamically adjust weights based on input-output timing relationships
- **Hebbian Learning Rule**: "Fire together, wire together" neural plasticity
- **Long-term Potentiation/Depression**: LTP/LTD mechanisms for memory strengthening and forgetting
- **Meta-learning Support**: Learn how to learn, adapt to different tasks

#### 4. Qwen3.5-0.8B Core Model

Integrating Qwen3.5-0.8B as the core language and reasoning engine:

- **Native Multimodal Support**: Unified text and image processing
- **Video Stream Processing**: Frame-by-frame decomposition, streaming analysis
- **Efficient Inference**: 0.8B parameters, suitable for edge deployment
- **Fine-tunable Architecture**: Support LoRA and other efficient fine-tuning methods

#### 5. Human-Brain-Like Memory System

Simulating hippocampal memory management mechanisms:

- **Neural Accumulative Growth**: Memory stored as neural connections accumulating over time
- **Hierarchical Memory Structure**: Short-term → Long-term → Permanent memory
- **Associative Retrieval**: Memory search based on semantic similarity and temporal association
- **Forgetting Mechanism**: Simulate human brain's forgetting curve, optimize storage space

#### 6. Infinite Knowledge Extension

Achieving infinite knowledge base expansion through Wikipedia API:

- **Real-time Knowledge Retrieval**: Get latest information on demand
- **Multi-language Support**: Support Chinese and English Wikipedia
- **Knowledge Fusion**: Integrate retrieval results with internal memory

### Project Highlights

#### ✅ Real Benchmark Testing

We promise to provide real, reproducible benchmark results:

| Benchmark | DigBrain Score | Baseline | Description |
|-----------|---------------|----------|-------------|
| MMLU | Measured | Baseline | Multi-task Language Understanding |
| HellaSwag | Measured | Baseline | Commonsense Reasoning |
| WinoGrande | Measured | Baseline | Coreference Resolution |
| ARC-Challenge | Measured | Baseline | AI Reasoning Challenge |
| TruthfulQA | Measured | Baseline | Factual Accuracy |
| GSM8K | Measured | Baseline | Mathematical Reasoning |

**Anti-Cheating Statement**:
- All tests use official evaluation scripts
- Test code is fully open source
- Provide complete test logs and configurations
- Support third-party reproduction verification

### Installation (Mac Platform)

#### System Requirements

- macOS 12.0+
- Python 3.10+
- 8GB+ RAM (16GB recommended)
- 20GB+ available disk space

#### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/ctz168/digbrain.git
cd digbrain

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download Qwen3.5-0.8B model
python scripts/download_model.py

# 5. Initialize memory storage
python scripts/init_memory.py

# 6. Run tests
pytest tests/

# 7. Start service
python -m digbrain.server
```

### Training Methods

#### Online Learning

```python
from digbrain import DigBrain

# Initialize system
brain = DigBrain()

# Enable online learning mode
brain.enable_online_learning(stdp_rate=0.01)

# Streaming input processing
for chunk in input_stream:
    response = brain.process(chunk)
    # STDP automatically updates weights
```

#### Offline Training

```bash
# Single module training
python -m digbrain.training.train_memory --data ./data/memory_train.json
python -m digbrain.training.train_stdp --episodes 1000

# Comprehensive multi-threaded training
python -m digbrain.training.train_all --config ./config/training.yaml
```

### API Usage

#### RESTful API

```python
import requests

# Text processing
response = requests.post(
    "http://localhost:8000/api/process",
    json={
        "input": "Hello, please introduce yourself",
        "stream": True,
        "search_wiki": True
    }
)

# Streaming response
for chunk in response.iter_content(chunk_size=None):
    print(chunk.decode(), end="")
```

### Web Interface

Start the web interface:

```bash
python -m digbrain.web.server
```

Visit http://localhost:3000 to use the graphical interface.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- Qwen Team - For the excellent Qwen3.5-0.8B model
- DeepSeek Team - Storage-computation separation architecture inspiration
- Neuroscience Research Community - Brain-inspired computing theoretical foundation

---

<div align="center">

**⭐ If this project helps you, please give it a star! ⭐**

</div>
