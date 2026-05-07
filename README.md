# PhilTrace - 哲学概念历时分析平台

PhilTrace 是一个基于自然语言处理和机器学习技术的哲学概念分析平台，旨在揭示哲学概念在历史长河中的语义演变规律。本项目结合了现代 NLP 技术（BERT 嵌入、UMAP 降维、HDBSCAN 聚类）与本地大语言模型（Ollama），为哲学研究提供了创新的数字人文工具。

## 🎯 核心功能

### 1. 概念漂移研究（Concept Drift Analysis）

通过对 110+ 篇经典哲学文献的语料库分析，追踪哲学概念在四个历史时期的语义演变：

- **Ancient（古代）**：公元前 6 世纪至公元 5 世纪（柏拉图、亚里士多德、斯多葛学派等）
- **Medieval（中世纪）**：5 世纪至 15 世纪（奥古斯丁、阿奎那等）
- **Modern（近代）**：15 世纪至 19 世纪末（笛卡尔、康德、黑格尔、尼采等）
- **Contemporary（当代）**：20 世纪至今（现象学、存在主义、分析哲学、后现代主义等）

**技术实现**：
- 使用 BERT 模型（sentence-transformers/all-MiniLM-L6-v2）生成上下文感知的句子嵌入
- UMAP 算法将高维语义向量投影到二维空间进行可视化
- HDBSCAN 密度聚类算法识别语义簇
- c-TF-IDF 提取每个语义簇的关键词特征

**可视化输出**：
- 语义空间散点图：展示概念在不同时期的分布模式
- 语义簇占比演化图：追踪各语义簇的历史消长
- 漂移趋势图：量化语义稳定性和概念使用频率
- 代表性引文：提供原文溯源和学术引用

### 2. 哲学对话（Philosophy Chat）

集成本地部署的大语言模型（通过 Ollama），提供专业的哲学讨论助手：

- **流式输出**：实时显示模型生成内容，提升交互体验
- **模型选择**：支持多种 Ollama 模型（Qwen2、Llama3、Mistral 等）
- **哲学风格增强**：内置专业哲学提示词，引导模型进行学术化讨论
- **对话历史**：保持上下文连贯性，支持深度对话

**哲学提示词特色**：
- 历史脉络把握：从古希腊到当代的完整哲学史知识
- 概念分析能力：精确定义、多维度解析、语义漂移追踪
- 文本解读方法：细读、论证结构分析、跨文本比较
- 引用规范：严格的学术引用格式和溯源要求
- 批判性思维：论证评估、谬误识别、多视角权衡

## 🏗️ 技术架构

### 前端（Frontend）
- **框架**：Streamlit 1.32+
- **UI 设计**：Bauhaus 风格（极简主义、几何形态、功能优先）
- **可视化**：Plotly Express（交互式图表）
- **状态管理**：Streamlit Session State

### 后端（Backend）
- **框架**：FastAPI 0.110+
- **异步支持**：asyncio + httpx
- **数据验证**：Pydantic v2
- **日志系统**：结构化日志 + 请求追踪（X-Request-ID）
- **错误处理**：统一异常处理和降级策略

### 核心算法引擎
- **嵌入模型**：sentence-transformers（BERT 系列）
- **降维算法**：UMAP（Uniform Manifold Approximation and Projection）
- **聚类算法**：HDBSCAN（Hierarchical Density-Based Spatial Clustering）
- **关键词提取**：c-TF-IDF（class-based TF-IDF）
- **设备支持**：自动检测 CUDA/MPS，支持 CPU 降级

### LLM 集成
- **服务**：Ollama（本地部署）
- **通信协议**：HTTP/SSE（Server-Sent Events）
- **超时控制**：600 秒（支持长时间推理）
- **流式传输**：支持实时 token 流式输出

### 数据管理
- **语料库**：110+ 篇哲学经典文献（TXT 格式）
- **缓存机制**：向量缓存 + 聚类结果缓存（Pickle 序列化）
- **缓存策略**：TTL 7 天，支持强制重算
- **数据持久化**：原子写入，防止缓存损坏

## 📁 项目结构

```
PhilTrace/
├── main.py                      # Streamlit 应用入口
├── requirements.txt             # 前端依赖
├── requirements_backend.txt     # 后端依赖
├── README.md                    # 项目文档
├── .gitignore                   # Git 忽略规则
│
├── ui/                          # 前端界面
│   ├── view_drift.py           # 概念漂移页面
│   ├── view_chat.py            # 对话页面
│   └── style.py                # Bauhaus UI 样式
│
├── backend/                     # 后端服务
│   ├── main.py                 # FastAPI 应用
│   ├── analysis_engine.py      # 概念分析引擎
│   ├── llm_service.py          # LLM 服务封装
│   ├── philosophy_prompts.py   # 哲学提示词
│   ├── data_models.py          # Pydantic 数据模型
│   ├── config.py               # 配置管理
│   └── logging_utils.py        # 日志工具
│
├── utils/                       # 工具模块
│   ├── api_client.py           # API 客户端
│   └── api_mock.py             # Mock 数据（开发/测试）
│
├── corpus/                      # 哲学文献语料库
│   ├── ancient/                # 古代哲学（27 篇）
│   ├── medieval/               # 中世纪哲学（10 篇）
│   ├── modern/                 # 近代哲学（30 篇）
│   └── contemporary/           # 当代哲学（43 篇）
│
├── tests/                       # 单元测试
│   ├── test_analysis_engine.py # 引擎测试
│   └── test_backend_api.py     # API 测试
│
├── data/                        # 运行时数据
│   └── cache/                  # 向量和聚类缓存
│
└── assets/                      # 静态资源
    └── logo.svg                # 项目 Logo
```

## 🚀 快速开始

### 环境要求

- Python 3.9+
- 8GB+ RAM（推荐 16GB）
- （可选）NVIDIA GPU + CUDA 11.0+（加速嵌入计算）
- （可选）Ollama 服务（用于对话功能）

### 1. 安装依赖

```bash
# 安装前端依赖
pip install -r requirements.txt

# 安装后端依赖
pip install -r requirements_backend.txt
```

### 2. 启动后端服务

```bash
uvicorn backend.main:app --reload
```

后端将在 `http://127.0.0.1:8000` 启动，可访问 `http://127.0.0.1:8000/docs` 查看 API 文档。

### 3. 启动前端应用

```bash
streamlit run main.py
```

前端将在 `http://localhost:8501` 启动。

### 4. （可选）安装 Ollama

对话功能需要本地运行 Ollama 服务：

```bash
# 安装 Ollama（参考 https://ollama.ai）
# 拉取模型
ollama pull qwen2:7b
ollama pull llama3:8b

# Ollama 默认运行在 http://127.0.0.1:11434
```

## 🎮 使用指南

### 概念漂移研究

1. 在左侧导航选择"概念漂移研究"
2. 输入要分析的哲学概念（英文），例如：
   - `reason`（理性）
   - `justice`（正义）
   - `truth`（真理）
   - `freedom`（自由）
   - `consciousness`（意识）
3. （可选）展开"算法参数微调"调整 UMAP 和 HDBSCAN 参数
4. 点击"开始分析"，等待处理完成（首次分析需要 30-60 秒）
5. 查看可视化结果：
   - 跨时期定义：每个时期的概念定义
   - 语义空间散点图：概念的语义分布
   - 语义簇占比演化：各语义簇的历史变化
   - 代表性语句：原文引用和来源

### 哲学对话

1. 在左侧导航选择"自由对话"
2. 在右侧设置面板选择模型和是否启用哲学风格
3. 在输入框输入问题，例如：
   - "请解释康德的先验综合判断"
   - "比较柏拉图和亚里士多德的形式理论"
   - "什么是现象学还原？"
4. 观察流式输出的回答
5. 继续追问以深化讨论

## ⚙️ 配置选项

通过环境变量自定义运行参数：

```bash
# 日志级别（DEBUG, INFO, WARNING, ERROR）
export PHILTRACE_LOG_LEVEL=INFO

# 启动时是否预加载分析引擎
export PHILTRACE_BOOTSTRAP_ENGINE=true

# 是否启用缓存
export PHILTRACE_ENABLE_CACHE=true

# 缓存有效期（秒）
export PHILTRACE_CACHE_TTL_SECONDS=604800

# API 超时时间（秒）
export PHILTRACE_API_TIMEOUT_SECONDS=600

# Ollama 服务地址
export PHILTRACE_OLLAMA_BASE_URL=http://127.0.0.1:11434

# 后端监听地址和端口
export PHILTRACE_HOST=0.0.0.0
export PHILTRACE_PORT=8000
```

## 🧪 运行测试

```bash
# 运行所有测试
python -m unittest discover -s tests -p "test_*.py" -v

# 运行特定测试
python -m unittest tests.test_analysis_engine -v
python -m unittest tests.test_backend_api -v
```

## 📊 性能优化

### 缓存机制
- **向量缓存**：首次分析后，概念的句子嵌入会被缓存，后续分析速度提升 10-20 倍
- **聚类缓存**：相同参数的聚类结果会被缓存，避免重复计算
- **缓存失效**：可通过"强制重算"选项绕过缓存

### GPU 加速
- 自动检测 CUDA（NVIDIA）或 MPS（Apple Silicon）
- GPU 加速可将嵌入计算速度提升 5-10 倍
- CPU 模式下使用优化的 NumPy 运算

### 超时控制
- 概念分析超时：600 秒（处理大量句子）
- LLM 对话超时：600 秒（支持长文本生成）
- 健康检查超时：10 秒


## 🔬 技术亮点

### 1. 语义漂移量化
通过聚类数量和分布计算语义稳定性指标，量化概念的历史演变程度。

### 2. 上下文感知嵌入
使用 BERT 模型生成上下文相关的句子嵌入，而非简单的词向量，更准确捕捉概念在不同语境中的含义。

### 3. 密度聚类
HDBSCAN 算法无需预设簇数，自动识别不同密度的语义簇，并标记噪声点。

### 4. 流式对话
采用 SSE（Server-Sent Events）实现 LLM 响应的实时流式传输，提升用户体验。

### 5. 降级策略
- 嵌入模型加载失败时，自动降级到随机向量（保证系统可用性）
- Ollama 服务不可用时，提供友好的错误提示
- 缓存损坏时，自动删除并重新计算

### 6. 请求追踪
每个 API 请求分配唯一的 `X-Request-ID`，便于日志追踪和问题排查。

## 📚 语料库说明

本项目语料库包含 110+ 篇经典哲学文献，涵盖：

**古代哲学**（27 篇）：
- 柏拉图：《理想国》《斐多篇》《会饮篇》等
- 亚里士多德：《形而上学》《尼各马可伦理学》《政治学》等
- 斯多葛学派：塞涅卡、爱比克泰德、马可·奥勒留
- 其他：赫拉克利特、卢克莱修、西塞罗等

**中世纪哲学**（10 篇）：
- 奥古斯丁：《忏悔录》《上帝之城》
- 托马斯·阿奎那：《神学大全》
- 安瑟伦、波爱修斯、迈蒙尼德等

**近代哲学**（30 篇）：
- 理性主义：笛卡尔、斯宾诺莎、莱布尼茨
- 经验主义：洛克、休谟、贝克莱
- 德国古典哲学：康德、黑格尔
- 其他：尼采、叔本华、卢梭、密尔等

**当代哲学**（43 篇）：
- 现象学：胡塞尔、海德格尔、梅洛-庞蒂
- 存在主义：萨特、加缪、波伏娃
- 分析哲学：罗素、维特根斯坦、奎因
- 后现代主义：福柯、德里达、德勒兹
- 其他：SEP 和 IEP 百科条目

## 🤝 贡献指南

本项目为学术研究和竞赛展示项目，欢迎提出改进建议。

## 📄 许可证

MIT License

## 🙏 致谢

- 语料来源：Project Gutenberg、Stanford Encyclopedia of Philosophy、Internet Encyclopedia of Philosophy
- 嵌入模型：sentence-transformers 项目
- LLM 服务：Ollama 项目
- UI 灵感：Bauhaus 设计运动

---

**PhilTrace** - 让哲学概念的历史演变可见可感
