# PhilTrace

**哲学概念语义漂移追踪平台** — 基于上下文嵌入和 LLM 诠释的哲学术语历时语义分析工具。

PhilTrace 从跨时期哲学语料中提取目标词的上下文用例，通过 ModernBERT 生成 token-level 嵌入，经 UMAP 降维和 HDBSCAN 聚类发现语义簇，最后由 LLM 为每个簇生成哲学诠释。

## 核心能力

- **Token-level 上下文嵌入** — ModernBERT-large (1024-dim) 精确提取目标词在句中的语境化表示
- **无监督语义聚类** — UMAP + HDBSCAN 自动发现词义分布
- **LLM 哲学诠释** — 对每个语义簇生成描述、关键词、所属哲学传统
- **历时对比** — 覆盖 Ancient / Medieval / Modern / Contemporary 四个时期
- **交互式前端** — Streamlit 可视化散点图、演化趋势、代表语句

## 技术栈

| 层级 | 技术 |
|------|------|
| Encoder | [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) |
| 降维 | UMAP |
| 聚类 | HDBSCAN |
| LLM | Ollama (qwen3 / gemma4 等本地模型) |
| 后端 | FastAPI + Uvicorn |
| 前端 | Streamlit + Plotly |
| 语料 | 110 篇哲学经典文本 (古代→当代) |

## 项目结构

```
PhilTrace/
├── backend/
│   ├── main.py                 # FastAPI 服务入口
│   ├── analysis_engine.py      # 分析引擎 (提取/编码/聚类)
│   ├── modernbert_encoder.py   # ModernBERT 编码器模块
│   ├── sense_interpreter.py    # LLM 语义诠释模块
│   ├── pipeline_config.py      # 管线配置加载器
│   ├── llm_service.py          # Ollama 对话服务
│   ├── config.py               # 环境配置
│   └── data_models.py          # API 数据模型
├── config/
│   └── pipeline.yaml           # 管线参数配置
├── corpus/                     # 哲学语料库 (4 个时期)
│   ├── ancient/                # 柏拉图、亚里士多德、斯多葛等
│   ├── medieval/               # 奥古斯丁、阿奎那等
│   ├── modern/                 # 笛卡尔、康德、黑格尔等
│   └── contemporary/           # SEP/IEP 条目
├── prompts/
│   └── sense_interpretation.md # LLM 诠释 prompt 模板
├── scripts/
│   └── run_demo.py             # 端到端 demo 脚本
├── tests/
│   ├── test_analysis_engine.py
│   └── test_sense_interpreter.py
├── ui/
│   ├── view_drift.py           # 概念漂移研究页面
│   └── view_chat.py            # 哲学对话页面
├── utils/
│   ├── api_client.py           # 后端 API 客户端
│   └── api_mock.py             # Mock 数据 (开发用)
├── main.py                     # Streamlit 入口
├── requirements.txt            # 前端依赖
└── requirements_backend.txt    # 后端依赖
```

## 快速开始

### 环境要求

- Python 3.10+
- [Ollama](https://ollama.ai/) (本地 LLM 推理)
- 约 2GB 磁盘空间 (ModernBERT-large 权重)

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/PhilTrace.git
cd PhilTrace

# 安装依赖
pip install -r requirements.txt
pip install -r requirements_backend.txt

# 拉取 LLM 模型
ollama pull qwen3:8b
```

### 运行 Demo

```bash
# 端到端 pipeline demo (编码 + 聚类 + LLM 诠释)
python -m scripts.run_demo --word justice --provider ollama

# 可选目标词: force, reason, cause, virtue, substance, justice
```

输出：
- 终端打印 pipeline 各阶段进度和统计
- `data/demo_output/demo_<word>.md` — Markdown 报告
- `data/interpretations/<word>_interpretations.json` — 结构化诠释结果

### 启动 Web 服务

```bash
# 后端 API
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000

# 前端 (新终端)
streamlit run main.py
```

## Pipeline 流程

```
语料库 (110 files)
    │
    ▼ 提取含目标词的句子
句子列表 + 时期标签
    │
    ▼ ModernBERT token-level encoding
1024-dim 向量矩阵
    │
    ▼ UMAP 降维 → 2D
    ▼ HDBSCAN 聚类
语义簇 (每簇 = 一个词义)
    │
    ▼ LLM 诠释 (Ollama)
结构化输出: 描述 / 关键词 / 哲学传统
    │
    ▼ 可视化 + 报告
```

## 配置

所有 pipeline 参数集中在 `config/pipeline.yaml`：

```yaml
encoder:
  model_name: "answerdotai/ModernBERT-large"
  subword_pooling: "mean"
  max_length: 8192

hdbscan:
  min_cluster_size: 15
  min_samples: 5

interpretation:
  provider: "ollama"
  ollama_model: "qwen3:8b"
  n_samples: 8
  sampling_strategy: "centroid_nearest"
```

## 测试

```bash
python -m unittest discover tests -v
```

## 语料库

覆盖西方哲学四个主要时期，共 110 篇文本：

| 时期 | 代表作者 | 文本数 |
|------|----------|--------|
| Ancient | Plato, Aristotle, Epictetus, Seneca | 27 |
| Medieval | Augustine, Aquinas, Anselm, Boethius | 10 |
| Modern | Descartes, Kant, Hegel, Nietzsche, Mill | 30 |
| Contemporary | SEP/IEP 哲学百科条目 | 43 |

## License

MIT
