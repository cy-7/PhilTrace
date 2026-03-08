"""
Mock API Module for PhilTrace Platform
=====================================
提供模拟的后端数据接口，用于前端开发和测试。
后期替换为真实 API 调用即可。
"""

import random
import time
from datetime import datetime
from typing import Dict, List, Any, Optional


# ============================================================================
# Mock Data Definitions - 可扩展的模拟数据仓库
# ============================================================================

# 哲学概念历史定义数据库（示例）
MOCK_CONCEPT_DEFINITIONS = {
    "Substance": {
        "Ancient": "The fundamental reality underlying all changes (Aristotle). Substance is that which exists independently, in contrast to accidents which depend on substance.",
        "Medieval": "The substratum supporting all attributes; God's creation as the primary substance with derived substances below (Aquinas).",
        "Modern": "Independent reality; Descartes' 'res' (thinking thing or extended thing). Spinoza's single infinite substance: God or Nature.",
        "Contemporary": "No fundamental entities; only patterns and relations matter. Ontological pluralism replaces substance ontology."
    },
    "Reason": {
        "Ancient": "Logos - the rational principle governing the cosmos. The faculty of logical deduction and understanding eternal truths.",
        "Medieval": "The image of God in man; subordinate to faith but capable of understanding divine truths through natural theology.",
        "Modern": "Autonomous human faculty for knowledge generation. Kant's critique of pure reason: limits and conditions of possibility.",
        "Contemporary": "Critiqued as Eurocentric, patriarchal. Procedural rationality vs. substantive rationality debates."
    },
    "Justice": {
        "Ancient": "Cosmic order (dike) and civic virtue. Aristotle's distributive and corrective justice.",
        "Medieval": "Divine justice; natural law as participation in eternal law (Aquinas).",
        "Modern": "Social contract theory. Rawls' justice as fairness: veil of ignorance behind original position.",
        "Contemporary": "Recognition theory, distributive vs. procedural. Intersectional justice concerns."
    }
}

# 默认定义（用于未收录的概念）
DEFAULT_DEFINITIONS = {
    "Ancient": "The ancient understanding of this concept emphasized its fundamental nature in the cosmos.",
    "Medieval": "Medieval scholars interpreted this concept through the lens of divine creation and natural order.",
    "Modern": "Modern philosophy re-conceptualized this idea in terms of human subjectivity and empirical knowledge.",
    "Contemporary": "Contemporary analysis deconstructs and pluralizes this concept across different traditions."
}

# 历史时期列表
HISTORICAL_PERIODS = ["Ancient", "Medieval", "Modern", "Contemporary"]

# 时期对应年份
PERIOD_YEARS = {
    "Ancient": "800 BCE - 500 CE",
    "Medieval": "500 CE - 1500 CE", 
    "Modern": "1500 CE - 1900 CE",
    "Contemporary": "1900 CE - Present"
}

# 语义漂移模拟数据生成器
def generate_semantic_drift_data(concept: str) -> List[Dict[str, Any]]:
    """生成语义漂移趋势图数据"""
    random.seed(hash(concept) % (2**31))
    
    periods = ["5th c. BCE", "1st c. CE", "5th c.", "10th c.", "15th c.", "18th c.", "19th c.", "20th c."]
    semantic_values = [random.uniform(0.3, 0.7) for _ in range(len(periods))]
    
    # 添加一些随机波动
    for i in range(1, len(semantic_values)):
        semantic_values[i] += random.uniform(-0.15, 0.15)
        semantic_values[i] = max(0.1, min(0.9, semantic_values[i]))
    
    return [
        {"period": p, "semantic_stability": v, "concept_usage": random.uniform(0.2, 1.0)}
        for p, v in zip(periods, semantic_values)
    ]

# 语义空间散点图数据生成器
def generate_semantic_space_data(concept: str) -> List[Dict[str, Any]]:
    """生成语义空间散点图数据"""
    random.seed(hash(concept) % (2**31) + 1)
    
    data = []
    for period in HISTORICAL_PERIODS:
        # 每个时期生成若干语义"邻居"概念
        n_points = random.randint(8, 15)
        for i in range(n_points):
            data.append({
                "period": period,
                "x": random.uniform(-1, 1),
                "y": random.uniform(-1, 1),
                "concept": f"Related Concept {i+1}",
                "distance": random.uniform(0.1, 0.8)
            })
    
    return data

# 示例句子数据库
MOCK_QUOTATIONS = {
    "Substance": [
        {"text": "Substance is that which is in itself and is conceived through itself.", "source": "Spinoza, Ethics", "year": "1677"},
        {"text": "We must determine what the word 'substance' can mean, if there is to be any sense in applying it.", "source": "Aristotle, Categories", "year": "~350 BCE"},
        {"text": "The notion of substance is the ultimate outcome of reflection.", "source": "Hegel, Science of Logic", "year": "1812"},
    ],
    "Reason": [
        {"text": "Reason is the law of the universe, the law that governs all existence.", "source": "Xunzi", "year": "~300 BCE"},
        {"text": "Two things fill the mind with ever-increasing wonder: the starry heavens above and the moral law within.", "source": "Kant, Critique of Practical Reason", "year": "1788"},
    ]
}

DEFAULT_QUOTATIONS = [
    {"text": "This concept has been fundamental to philosophical inquiry throughout history.", "source": "Historical Analysis", "year": "Various"},
    {"text": "Various interpretations have emerged as understanding evolved across traditions.", "source": "Comparative Philosophy", "year": "Ongoing"},
]


# ============================================================================
# Public Mock API Functions
# ============================================================================

def mock_get_concept_analysis(
    concept: str, 
    periods: Optional[List[str]] = None,
    include_examples: bool = True
) -> Dict[str, Any]:
    """
    获取哲学概念的历时分析数据
    
    Args:
        concept: 哲学概念名称（如 "Substance", "Justice"）
        periods: 可选，指定分析的历史时期
        include_examples: 是否包含典型例句
    
    Returns:
        Dict containing:
        - concept: 概念名称
        - periods: 分析的时期列表
        - definitions: 各时期定义
        - drift_data: 语义漂移趋势数据
        - scatter_data: 语义空间散点数据
        - quotations: 典型例句列表
    
    NOTE: 此为 Mock 函数，后期替换为真实 API 调用：
        真实调用示例：
        >>> import requests
        >>> response = requests.post(
        ...     "https://api.philtrace.com/v1/concept/analyze",
        ...     json={"concept": concept, "periods": periods}
        ... )
        >>> return response.json()
    """
    # 模拟网络延迟
    time.sleep(0.3)
    
    # 获取定义
    if concept in MOCK_CONCEPT_DEFINITIONS:
        definitions = MOCK_CONCEPT_DEFINITIONS[concept].copy()
    else:
        definitions = DEFAULT_DEFINITIONS.copy()
    
    # 过滤指定时期
    if periods:
        definitions = {k: v for k, v in definitions.items() if k in periods}
    
    # 生成可视化数据
    drift_data = generate_semantic_drift_data(concept)
    scatter_data = generate_semantic_space_data(concept)
    
    # 获取例句
    if include_examples:
        quotations = MOCK_QUOTATIONS.get(concept, DEFAULT_QUOTATIONS)
    else:
        quotations = []
    
    return {
        "concept": concept,
        "periods": list(definitions.keys()),
        "definitions": definitions,
        "period_years": {k: PERIOD_YEARS[k] for k in definitions.keys()},
        "drift_data": drift_data,
        "scatter_data": scatter_data,
        "quotations": quotations,
        "analysis_timestamp": datetime.now().isoformat(),
        "confidence_score": random.uniform(0.75, 0.98)
    }


def mock_get_historical_periods() -> List[Dict[str, str]]:
    """
    获取可用的历史时期列表
    
    Returns:
        List of dicts with 'id' and 'name' keys
    
    NOTE: 真实 API 应返回数据库中的所有时期配置
    """
    return [
        {"id": "ancient", "name": "Ancient (800 BCE - 500 CE)", "order": 1},
        {"id": "medieval", "name": "Medieval (500 CE - 1500 CE)", "order": 2},
        {"id": "modern", "name": "Modern (1500 CE - 1900 CE)", "order": 3},
        {"id": "contemporary", "name": "Contemporary (1900 CE - Present)", "order": 4},
    ]


def mock_chat_response(
    message: str,
    model: str = "Qwen-7B",
    lora: Optional[str] = None,
    stream: bool = False
) -> Dict[str, Any]:
    """
    模拟哲学对话响应
    
    Args:
        message: 用户输入的消息
        model: 选择的基座模型
        lora: 选择的 LoRA 适配器（可选）
        stream: 是否流式输出
    
    Returns:
        Dict containing response and metadata
    
    NOTE: 真实 API 调用示例：
        >>> response = requests.post(
        ...     "https://api.philtrace.com/v1/chat",
        ...     json={
        ...         "message": message,
        ...         "model": model,
        ...         "adapter": lora,
        ...         "stream": stream
        ...     }
        ... )
        >>> return response.json()
    """
    # 模拟处理延迟
    time.sleep(0.5 if not stream else 0.1)
    
    # 模拟回复
    mock_responses = [
        f"这是一个关于 \"{message[:20]}...\" 的深刻哲学问题。让我从多个传统来分析...",
        f"从分析哲学的角度来看，这个概念需要首先进行语言层面的澄清...",
        f"您提到的这一点让我联想到现象学传统中的相关讨论...",
    ]
    
    response_text = random.choice(mock_responses)
    
    # 添加 LoRA 相关上下文
    if lora:
        response_text += f"\n\n[使用 {lora} 适配器进行风格化输出]"
    
    return {
        "response": response_text,
        "model": model,
        "adapter": lora,
        "token_count": random.randint(50, 200),
        "response_time": random.uniform(0.5, 2.0)
    }


def mock_get_available_models() -> List[Dict[str, Any]]:
    """
    获取可用的模型列表
    
    Returns:
        List of available models with their configurations
    """
    return [
        {
            "id": "qwen-7b",
            "name": "Qwen-7B",
            "provider": "Qwen Team",
            "context_length": 8192,
            "description": "通识能力强的中英文基座模型"
        },
        {
            "id": "llama-3-8b",
            "name": "Llama-3 8B",
            "provider": "Meta",
            "context_length": 8192,
            "description": "开源多语言基座模型"
        },
        {
            "id": "chatglm-6b",
            "name": "ChatGLM-6B",
            "provider": "Zhipu AI",
            "context_length": 8192,
            "description": "中文对话增强模型"
        }
    ]


def mock_get_available_loras() -> List[Dict[str, str]]:
    """
    获取可用的 LoRA 适配器列表
    
    Returns:
        List of available LoRA adapters
    """
    return [
        {"id": "spinoza", "name": "Spinoza-LoRA", "description": "斯宾诺莎伦理学风格"},
        {"id": "kant", "name": "Kant-LoRA", "description": "康德批判哲学风格"},
        {"id": "hegel", "name": "Hegel-LoRA", "description": "黑格尔辩证法风格"},
        {"id": "confucian", "name": "Confucian-LoRA", "description": "儒家经典诠释风格"},
        {"id": "analytic", "name": "Analytic-LoRA", "description": "分析哲学论证风格"},
    ]


def mock_search_concepts(query: str) -> List[Dict[str, str]]:
    """
    搜索哲学概念
    
    Args:
        query: 搜索关键词
    
    Returns:
        List of matching concepts
    
    NOTE: 真实 API 应调用全文搜索引擎
    """
    all_concepts = list(MOCK_CONCEPT_DEFINITIONS.keys())
    # 简单模糊匹配
    matches = [c for c in all_concepts if query.lower() in c.lower()]
    
    # 如果没有精确匹配，返回一些推荐
    if not matches:
        matches = random.sample(all_concepts, min(3, len(all_concepts)))
    
    return [{"id": m.lower(), "name": m} for m in matches]
