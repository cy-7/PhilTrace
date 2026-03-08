"""Concept drift study page."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.api_client import AnalysisResult, get_client_with_fallback

PERIOD_ORDER = ["Ancient", "Medieval", "Modern", "Contemporary"]
PERIOD_COLORS = {
    "Ancient": "#E53935",
    "Medieval": "#FDD835",
    "Modern": "#1E88E5",
    "Contemporary": "#212121",
}


def render_page_header() -> None:
    st.markdown(
        """
        <div style="margin-bottom:1rem;">
            <h1 class="phil-title">Concept Drift Studio</h1>
            <p class="phil-subtitle">
                输入一个哲学概念，追踪其在古代、中世纪、近代、当代文本中的语义变化。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_concept_input() -> str:
    st.markdown(
        """
        <div class="concept-input-section fade-in">
          <h3 style="margin:0 0 0.4rem 0;">研究入口</h3>
          <p style="margin:0; color:var(--phil-text-secondary);">
            示例：Substance / Reason / Justice
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return st.text_input(
        label="概念输入",
        placeholder="请输入概念，例如 Reason",
        label_visibility="collapsed",
        key="concept_search",
    ).strip()


def render_algorithm_tuning() -> Dict[str, Any]:
    with st.expander("算法参数微调", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            min_cluster_size = st.slider("最小簇大小", 5, 100, 20, step=5)
            n_neighbors = st.slider("UMAP 邻居数", 5, 50, 15, step=1)
        with c2:
            min_samples = st.slider("最小样本数", 1, 20, 5, step=1)
            force_recompute = st.checkbox("强制重算（忽略缓存）", value=False)

        return {
            "umap_params": {
                "n_neighbors": n_neighbors,
                "min_dist": 0.1,
                "n_components": 2,
                "metric": "cosine",
            },
            "hdbscan_params": {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "metric": "euclidean",
                "cluster_selection_method": "eom",
            },
            "force_recompute": force_recompute,
        }


def render_period_cards(definitions: Dict[str, str], period_years: Dict[str, str]) -> None:
    st.markdown('<div class="periods-grid">', unsafe_allow_html=True)

    for period in PERIOD_ORDER:
        if period not in definitions:
            continue

        color = PERIOD_COLORS.get(period, "#212121")
        st.markdown(
            f"""
            <div class="phil-card period-card fade-in">
              <h3>
                <span>{period}</span>
                <span class="period-badge" style="background:#FFFFFF;color:{color};">{period_years.get(period, '')}</span>
              </h3>
              <p>{definitions[period]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df.copy()


def render_semantic_scatter_chart(scatter_data: pd.DataFrame) -> None:
    st.markdown('<div class="chart-container"><h4 class="chart-title">语义空间散点图</h4>', unsafe_allow_html=True)

    df = _safe_df(scatter_data)
    if df.empty:
        st.info("暂无散点数据。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for col in ["x", "y", "period"]:
        if col not in df.columns:
            st.warning(f"散点数据缺少字段: {col}")
            st.markdown("</div>", unsafe_allow_html=True)
            return

    if "cluster_keywords" not in df.columns:
        df["cluster_keywords"] = ""

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="period",
        color_discrete_map=PERIOD_COLORS,
        hover_data={
            "text": True,
            "source": True,
            "label": True,
            "cluster_keywords": True,
            "x": False,
            "y": False,
        },
    )

    fig.update_traces(marker={"size": 9, "opacity": 0.85, "line": {"width": 1, "color": "#212121"}})
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis={"title": "UMAP-1", "showgrid": True, "gridcolor": "#E0E0E0", "zeroline": False},
        yaxis={"title": "UMAP-2", "showgrid": True, "gridcolor": "#E0E0E0", "zeroline": False},
        legend={"orientation": "h", "y": 1.06, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_evolution_chart(scatter_data: pd.DataFrame, cluster_stats: List[Dict[str, Any]]) -> None:
    st.markdown('<div class="chart-container"><h4 class="chart-title">语义簇占比演化</h4>', unsafe_allow_html=True)

    df = _safe_df(scatter_data)
    if df.empty or "period" not in df.columns or "label" not in df.columns:
        st.info("暂无可绘制演化图的数据。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    cluster_counts = df.groupby(["period", "label"]).size().reset_index(name="count")
    totals = cluster_counts.groupby("period")["count"].transform("sum")
    cluster_counts["percentage"] = (cluster_counts["count"] / totals) * 100

    label_to_keywords = {
        int(item["cluster_id"]): ", ".join(item.get("top_keywords", [])[:3])
        for item in cluster_stats
        if int(item.get("cluster_id", -1)) >= 0
    }
    cluster_counts["cluster_label"] = cluster_counts["label"].apply(
        lambda x: f"C{x}: {label_to_keywords.get(int(x), '')}" if int(x) >= 0 else "Noise"
    )

    fig = px.area(
        cluster_counts,
        x="period",
        y="percentage",
        color="cluster_label",
        category_orders={"period": PERIOD_ORDER},
    )
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis={"title": "历史时期", "showgrid": False},
        yaxis={"title": "占比(%)", "range": [0, 100], "showgrid": True, "gridcolor": "#E0E0E0"},
        legend={"orientation": "v"},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_drift_trend_chart(drift_data: pd.DataFrame) -> None:
    st.markdown('<div class="chart-container"><h4 class="chart-title">稳定性与使用频率趋势</h4>', unsafe_allow_html=True)

    df = _safe_df(drift_data)
    if df.empty or "period" not in df.columns or "semantic_stability" not in df.columns:
        st.info("暂无漂移趋势数据。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["period"],
            y=df["semantic_stability"],
            mode="lines+markers",
            line={"color": "#1E88E5", "width": 3},
            marker={"size": 8, "color": "#1E88E5"},
            name="语义稳定性",
            hovertemplate="%{y:.1%}<extra></extra>",
        )
    )

    if "concept_usage" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["period"],
                y=df["concept_usage"],
                mode="lines+markers",
                line={"color": "#E53935", "width": 2, "dash": "dot"},
                marker={"size": 7, "color": "#E53935"},
                name="概念使用频率",
                hovertemplate="%{y:.1%}<extra></extra>",
            )
        )

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis={"title": "历史时期", "categoryorder": "array", "categoryarray": PERIOD_ORDER},
        yaxis={"title": "比例", "range": [0, 1.05], "showgrid": True, "gridcolor": "#E0E0E0"},
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_cluster_stats(cluster_stats: List[Dict[str, Any]]) -> None:
    if not cluster_stats:
        return

    st.markdown("### 语义簇统计")
    cols = st.columns(min(len(cluster_stats), 4))

    for idx, stat in enumerate(cluster_stats[:8]):
        keywords = ", ".join(stat.get("top_keywords", [])[:3]) or "-"
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-value">C{stat['cluster_id']}</div>
                  <div class="metric-label">样本数: {stat['count']}</div>
                  <div class="metric-label">主导时期: {stat['dominant_period']}</div>
                  <div class="metric-label">关键词: {keywords}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_quotations(quotations: List[Dict[str, str]]) -> None:
    with st.expander("查看代表语句", expanded=True):
        if not quotations:
            st.info("当前没有代表语句。")
            return

        for i, quote in enumerate(quotations):
            st.markdown(
                f"""
                <div class="quote-card fade-in" style="animation-delay:{i * 0.04}s;">
                  <p class="quote-text">"{quote.get('text', '')}"</p>
                  <p class="quote-source">{quote.get('source', 'Unknown')} {f"({quote.get('period', '')})" if quote.get('period') else ""}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_analysis_metrics(result: AnalysisResult) -> None:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-value">{len(result.periods)}</div>
              <div class="metric-label">分析时期</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-value">{result.confidence_score:.0%}</div>
              <div class="metric-label">置信度</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-value">{len(result.quotations)}</div>
              <div class="metric-label">引用条目</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        n_points = 0 if result.scatter_data.empty else len(result.scatter_data)
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-value">{n_points}</div>
              <div class="metric-label">语义点 / {result.processing_time_ms / 1000:.1f}s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_drift_view() -> None:
    render_page_header()
    tuning = render_algorithm_tuning()
    concept = render_concept_input()

    if not concept:
        st.markdown(
            """
            <div class="welcome-message">
                <div class="welcome-title">输入一个概念开始研究</div>
                <div class="welcome-hint">建议先试：Reason、Justice、Substance。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    with st.spinner("正在执行概念漂移分析..."):
        client = get_client_with_fallback()
        try:
            result = client.get_concept_analysis(
                word=concept,
                umap_params=tuning["umap_params"],
                hdbscan_params=tuning["hdbscan_params"],
                force_recompute=tuning["force_recompute"],
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"分析失败：{exc}")
            st.info("请确认后端已运行：uvicorn backend.main:app --reload")
            return

    st.markdown("---")
    st.markdown('<h2 class="phil-section-title">跨时期定义</h2>', unsafe_allow_html=True)
    render_period_cards(result.definitions, result.period_years)
    render_analysis_metrics(result)

    if result.cluster_stats:
        render_cluster_stats(result.cluster_stats)

    st.markdown("---")
    st.markdown('<h2 class="phil-section-title">可视化分析</h2>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        render_semantic_scatter_chart(result.scatter_data)
    with right:
        render_evolution_chart(result.scatter_data, result.cluster_stats)

    render_drift_trend_chart(result.drift_data)

    st.markdown("---")
    st.markdown('<h2 class="phil-section-title">代表语句</h2>', unsafe_allow_html=True)
    render_quotations(result.quotations)

    with st.expander("开发者信息", expanded=False):
        st.markdown(
            f"""
            - 概念：`{result.concept}`
            - 模型：`{(result.model_info or {}).get('name', 'Mock/Fallback')}`
            - 处理耗时：`{result.processing_time_ms}ms`
            - 语义点数量：`{len(result.scatter_data)}`
            - 聚类数量：`{len(result.cluster_stats)}`
            - 时间戳：`{result.analysis_timestamp}`
            """
        )

        if not result.scatter_data.empty:
            st.dataframe(result.scatter_data.head(10), use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(page_title="PhilTrace - Concept Drift", layout="wide")
    from ui.style import apply_all_styles

    apply_all_styles()
    render_drift_view()
