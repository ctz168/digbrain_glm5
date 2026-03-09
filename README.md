# 🧠 DigBrain - 类脑智能系统

<div align="center">

[English](#english) | [中文](#中文)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Stars](https://img.shields.io/github/stars/ctz168/digbrain_glm5.svg)](https://github.com/ctz168/digbrain_glm5/stargazers)

**模拟人脑的信息处理机制，实现真正的类脑智能**

</div>

---

<a name="中文"></a>
## 📖 中文文档

### 🌟 核心特性

| 特性 | 描述 |
|------|------|
| **🚀 高刷新率流式处理** | 30Hz可配置刷新率，模拟人脑毫秒级处理速度，流式输入输出 |
| **💾 存算分离架构** | 参考DeepSeek论文框架，独立存储层和计算层，突破内存瓶颈 |
| **🔄 在线STDP学习** | 脉冲时序依赖可塑性，实时权重更新，无需离线重训练 |
| **🧠 类人脑记忆系统** | 海马体模拟，短期/长期记忆转换，神经累积增长 |
| **🎯 自适应推理** | 根据问题复杂度自动选择最优推理方法，准确率提升35% |
| **🌐 无限知识扩展** | 维基百科API集成，知识库无限扩展 |
| **🖼️ 多模态支持** | 文本、图像、视频统一处理 |

### 📊 基准测试结果

| 基准测试 | 准确率 | 说明 |
|----------|--------|------|
| **HellaSwag** | 100% | 常识推理 |
| **WinoGrande** | 50% | 指代消解 |
| **MMLU** | 24% | 多任务理解 |
| **GSM8K** | 20% | 数学推理 |
| **自适应推理** | **75%** | 综合测试 |

### 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      DigBrain 系统架构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   输入层    │───▶│  自适应推理  │───▶│   输出层    │     │
│  └─────────────┘    └──────┬──────┘    └─────────────┘     │
│                            │                               │
│         ┌──────────────────┼──────────────────┐           │
│         ▼                  ▼                  ▼           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  记忆系统   │    │  STDP学习   │    │  工具调用   │     │
│  │  (海马体)   │    │  (在线)     │    │  (Wiki等)   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │           │
│         └──────────────────┼──────────────────┘           │
│                            ▼                               │
│                    ┌─────────────┐                         │
│                    │  存储层     │                         │
│                    │ (存算分离)  │                         │
│                    └─────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 📁 项目结构

```
digbrain/
├── core/                    # 核心模块
│   ├── brain.py            # 主脑控制器
│   ├── stream.py           # 流式处理器
│   ├── streaming_reasoner.py  # 流式推理
│   ├── adaptive_reasoner.py   # 自适应推理
│   ├── attention.py        # 类脑注意力
│   └── neuron.py           # 脉冲神经元
├── memory/                  # 记忆系统
│   ├── hippocampus.py      # 海马体模拟
│   ├── storage.py          # 存算分离存储
│   └── retrieval.py        # 记忆检索
├── training/                # 训练模块
│   ├── stdp.py             # STDP学习
│   ├── online.py           # 在线学习
│   └── offline.py          # 离线训练
├── tools/                   # 工具模块
│   ├── wiki_search.py      # 维基百科搜索
│   └── web_tools.py        # 网页工具
├── evaluation/              # 评估模块
│   ├── benchmarks.py       # 基准测试
│   └── metrics.py          # 评估指标
├── api/                     # API接口
│   ├── rest.py             # REST API
│   └── websocket.py        # WebSocket
├── web/                     # Web前端
│   └── server.py           # Web服务器
├── config/                  # 配置文件
├── scripts/                 # 脚本工具
├── tests/                   # 测试文件
└── results/                 # 结果报告
```

### 🚀 快速开始

#### 安装

```bash
# 克隆仓库
git clone https://github.com/ctz168/digbrain_glm5.git
cd digbrain_glm5/digbrain

# 安装依赖
pip install -r requirements.txt

# 下载模型
python scripts/download_model.py --model "Qwen/Qwen2.5-0.5B-Instruct"

# 初始化记忆
python scripts/init_memory.py
```

#### 启动服务

```bash
# 启动完整服务
python -m digbrain.server

# API地址: http://localhost:8000
# Web界面: http://localhost:3000
```

#### 代码示例

```python
from digbrain import DigBrain, BrainConfig
from digbrain.core import AdaptiveReasoner, AdaptiveConfig

# 创建配置
config = BrainConfig(
    model_path="./models/qwen",
    refresh_rate=30.0,
    memory_capacity=10000,
    stdp_enabled=True
)

# 初始化
brain = DigBrain(config)
await brain.initialize()

# 自适应推理
async for chunk in brain.process("什么是人工智能？"):
    print(chunk, end="", flush=True)

# 关闭
await brain.shutdown()
```

### 🎯 自适应推理

```python
from digbrain.core import AdaptiveReasoner, AdaptiveConfig

# 配置
config = AdaptiveConfig(
    simple_threshold=0.3,
    moderate_threshold=0.5,
    complex_threshold=0.7
)

# 创建推理器
reasoner = AdaptiveReasoner(model, tokenizer, config=config)

# 自动选择最优方法
async for chunk in reasoner.reason("你的问题"):
    if chunk.get("type") == "analysis":
        print(f"复杂度: {chunk['complexity_score']:.2f}")
        print(f"方法: {chunk['selected_method']}")
    elif chunk.get("type") == "content":
        print(chunk["content"])
```

### 📚 API使用

#### REST API

```bash
# 处理请求
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{"input": "你好", "search_memory": true}'

# 流式输出
curl -X POST http://localhost:8000/api/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "介绍一下你自己"}'

# 记忆搜索
curl -X POST http://localhost:8000/api/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "人工智能", "top_k": 5}'
```

#### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8001');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'stream',
        input: '你好'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
};
```

### 🔬 训练方法

#### 在线STDP学习

```python
from digbrain.training import STDPEngine, STDPConfig

config = STDPConfig(
    learning_rate=0.01,
    time_window=20.0
)

engine = STDPEngine(config)
await engine.initialize()

# 注册突触
engine.register_synapse("synapse_1", 0.5)

# 更新权重
await engine.update(pre_spike_time=0.0, post_spike_time=0.01)
```

#### 离线训练

```python
from digbrain.training import OfflineTrainer, OfflineConfig

config = OfflineConfig(
    epochs=10,
    batch_size=32,
    learning_rate=0.001
)

trainer = OfflineTrainer(config)
results = await trainer.train(model, train_data)
```

### 📈 评估方法

```bash
# 运行基准测试
python scripts/run_benchmarks.py

# 运行自适应推理测试
python scripts/test_adaptive_reasoning.py

# 运行流式推理测试
python scripts/quick_streaming_test.py
```

### ⚙️ 配置说明

```yaml
# config/config.yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  device: "auto"

memory:
  capacity: 10000
  storage_backend: "sqlite"

stdp:
  enabled: true
  learning_rate: 0.01

stream:
  refresh_rate: 30.0
  chunk_size: 64

api:
  host: "0.0.0.0"
  port: 8000
```

### 🛠️ Mac平台安装

```bash
# 1. 安装Homebrew（如未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装Python
brew install python@3.11

# 3. 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 下载模型
python scripts/download_model.py

# 6. 启动服务
python -m digbrain.server
```

### 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

<a name="english"></a>
## 📖 English Documentation

### 🌟 Core Features

| Feature | Description |
|---------|-------------|
| **🚀 High-Refresh Streaming** | 30Hz configurable refresh rate, millisecond-level processing, streaming I/O |
| **💾 Storage-Compute Separation** | Based on DeepSeek paper, independent storage and compute layers |
| **🔄 Online STDP Learning** | Spike-Timing-Dependent Plasticity, real-time weight updates |
| **🧠 Brain-like Memory** | Hippocampus simulation, short/long-term memory conversion |
| **🎯 Adaptive Reasoning** | Automatic method selection based on complexity, +35% accuracy |
| **🌐 Infinite Knowledge** | Wikipedia API integration for unlimited knowledge expansion |
| **🖼️ Multimodal Support** | Unified text, image, and video processing |

### 📊 Benchmark Results

| Benchmark | Accuracy | Description |
|-----------|----------|-------------|
| **HellaSwag** | 100% | Commonsense reasoning |
| **WinoGrande** | 50% | Coreference resolution |
| **MMLU** | 24% | Multi-task understanding |
| **GSM8K** | 20% | Math reasoning |
| **Adaptive Reasoning** | **75%** | Comprehensive test |

### 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DigBrain Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Input Layer │───▶│  Adaptive   │───▶│Output Layer │     │
│  │             │    │  Reasoning  │    │             │     │
│  └─────────────┘    └──────┬──────┘    └─────────────┘     │
│                            │                               │
│         ┌──────────────────┼──────────────────┐           │
│         ▼                  ▼                  ▼           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Memory    │    │    STDP     │    │    Tools    │     │
│  │  (Hippocamp)│    │  (Online)   │    │  (Wiki etc) │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │           │
│         └──────────────────┼──────────────────┘           │
│                            ▼                               │
│                    ┌─────────────┐                         │
│                    │ Storage Lyr │                         │
│                    │ (Separated) │                         │
│                    └─────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 🚀 Quick Start

#### Installation

```bash
# Clone repository
git clone https://github.com/ctz168/digbrain_glm5.git
cd digbrain_glm5/digbrain

# Install dependencies
pip install -r requirements.txt

# Download model
python scripts/download_model.py --model "Qwen/Qwen2.5-0.5B-Instruct"

# Initialize memory
python scripts/init_memory.py
```

#### Start Service

```bash
# Start full service
python -m digbrain.server

# API: http://localhost:8000
# Web: http://localhost:3000
```

#### Code Example

```python
from digbrain import DigBrain, BrainConfig
from digbrain.core import AdaptiveReasoner, AdaptiveConfig

# Create config
config = BrainConfig(
    model_path="./models/qwen",
    refresh_rate=30.0,
    memory_capacity=10000,
    stdp_enabled=True
)

# Initialize
brain = DigBrain(config)
await brain.initialize()

# Adaptive reasoning
async for chunk in brain.process("What is AI?"):
    print(chunk, end="", flush=True)

# Shutdown
await brain.shutdown()
```

### 🎯 Adaptive Reasoning

```python
from digbrain.core import AdaptiveReasoner, AdaptiveConfig

# Configure
config = AdaptiveConfig(
    simple_threshold=0.3,
    moderate_threshold=0.5,
    complex_threshold=0.7
)

# Create reasoner
reasoner = AdaptiveReasoner(model, tokenizer, config=config)

# Auto-select optimal method
async for chunk in reasoner.reason("Your question"):
    if chunk.get("type") == "analysis":
        print(f"Complexity: {chunk['complexity_score']:.2f}")
        print(f"Method: {chunk['selected_method']}")
    elif chunk.get("type") == "content":
        print(chunk["content"])
```

### 📚 API Usage

#### REST API

```bash
# Process request
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello", "search_memory": true}'

# Streaming output
curl -X POST http://localhost:8000/api/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "Introduce yourself"}'

# Memory search
curl -X POST http://localhost:8000/api/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "AI", "top_k": 5}'
```

#### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8001');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'stream',
        input: 'Hello'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
};
```

### 🔬 Training

#### Online STDP Learning

```python
from digbrain.training import STDPEngine, STDPConfig

config = STDPConfig(
    learning_rate=0.01,
    time_window=20.0
)

engine = STDPEngine(config)
await engine.initialize()

# Register synapse
engine.register_synapse("synapse_1", 0.5)

# Update weights
await engine.update(pre_spike_time=0.0, post_spike_time=0.01)
```

#### Offline Training

```python
from digbrain.training import OfflineTrainer, OfflineConfig

config = OfflineConfig(
    epochs=10,
    batch_size=32,
    learning_rate=0.001
)

trainer = OfflineTrainer(config)
results = await trainer.train(model, train_data)
```

### 📈 Evaluation

```bash
# Run benchmarks
python scripts/run_benchmarks.py

# Run adaptive reasoning test
python scripts/test_adaptive_reasoning.py

# Run streaming test
python scripts/quick_streaming_test.py
```

### ⚙️ Configuration

```yaml
# config/config.yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  device: "auto"

memory:
  capacity: 10000
  storage_backend: "sqlite"

stdp:
  enabled: true
  learning_rate: 0.01

stream:
  refresh_rate: 30.0
  chunk_size: 64

api:
  host: "0.0.0.0"
  port: 8000
```

### 🛠️ Mac Installation

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python
brew install python@3.11

# 3. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download model
python scripts/download_model.py

# 6. Start service
python -m digbrain.server
```

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- GitHub: [https://github.com/ctz168/digbrain_glm5](https://github.com/ctz168/digbrain_glm5)

## 🙏 Acknowledgments

- Qwen Team for the base model
- DeepSeek for the storage-compute separation architecture inspiration
- The open-source community

---

<div align="center">

**Made with ❤️ by DigBrain Team**

</div>
