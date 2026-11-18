"""
PhilTrace End-to-End Demo Script.

Runs the full pipeline for a target word:
1. Extract usages from corpus
2. Encode with ModernBERT
3. UMAP + HDBSCAN clustering
4. LLM interpretation of each cluster
5. Generate markdown demo report

Usage:
    python -m scripts.run_demo --word force
    python -m scripts.run_demo --word cause --provider ollama
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.analysis_engine import AnalysisEngine, PERIODS, PERIOD_YEARS
from backend.logging_utils import get_logger
from backend.modernbert_encoder import ModernBERTEncoder
from backend.pipeline_config import get_pipeline_config
from backend.sense_interpreter import (
    SenseInterpretation,
    create_interpreter,
    save_interpretations,
)

logger = get_logger(__name__)


def run_demo(
    target_word: str,
    provider: str = "ollama",
    force_recompute: bool = False,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Run the full PhilTrace pipeline for a target word and generate a demo report.

    Returns the path to the generated markdown report.
    """
    output_dir = output_dir or Path(__file__).parent.parent / "data" / "demo_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PhilTrace Demo: '{target_word}'")
    print(f"{'='*60}\n")

    # ========================================================================
    # Step 1: Initialize engine and extract + encode
    # ========================================================================
    print("[1/5] Initializing analysis engine...")
    t0 = time.time()
    engine = AnalysisEngine(use_cache=not force_recompute)
    print(f"      Encoder: {engine.encoder.model_name_loaded or 'fallback'}")
    print(f"      Device: {engine.encoder.device}")
    print(f"      Hidden size: {engine.encoder.hidden_size}")
    print(f"      Corpus files: {sum(len(v) for v in engine.corpus_files.values())}")
    t_init = time.time() - t0
    print(f"      Init time: {t_init:.1f}s\n")

    # ========================================================================
    # Step 2: Get vectors (extract sentences + encode)
    # ========================================================================
    print(f"[2/5] Extracting and encoding usages of '{target_word}'...")
    t0 = time.time()
    vectors_df = engine.get_vectors(target_word, force_recompute=force_recompute)
    t_encode = time.time() - t0

    if vectors_df.empty:
        print(f"      ERROR: No usages found for '{target_word}' in corpus!")
        print("      Try a different word (e.g., force, cause, reason, justice)")
        return output_dir / "empty.md"

    print(f"      Found {len(vectors_df)} usages")
    print(f"      Period distribution:")
    for period in PERIODS:
        count = len(vectors_df[vectors_df["period"] == period])
        print(f"        {period}: {count}")

    # Embedding stats
    vectors = vectors_df["vector"].tolist()
    stats = engine.encoder.compute_embedding_stats(vectors)
    print(f"      Vector shape: {stats.get('shape', 'N/A')}")
    print(f"      Norm mean/std: {stats.get('norm_mean', 0):.4f} / {stats.get('norm_std', 0):.4f}")
    print(f"      Encoding time: {t_encode:.1f}s\n")

    # ========================================================================
    # Step 3: UMAP + HDBSCAN clustering
    # ========================================================================
    print("[3/5] Running UMAP + HDBSCAN clustering...")
    t0 = time.time()
    clustered_df = engine.cluster(vectors_df)
    t_cluster = time.time() - t0

    labels = clustered_df["label"].values
    n_clusters = len(set(int(l) for l in labels if int(l) != -1))
    n_noise = int((labels == -1).sum())
    print(f"      Clusters found: {n_clusters}")
    print(f"      Noise points: {n_noise}")
    print(f"      Clustering time: {t_cluster:.1f}s\n")

    # ========================================================================
    # Step 4: LLM interpretation
    # ========================================================================
    print(f"[4/5] Running LLM interpretation (provider: {provider})...")
    t0 = time.time()

    interpreter = create_interpreter(provider=provider)
    interpretations = interpreter.interpret_all_clusters(
        target_word=target_word,
        clustered_df=clustered_df,
    )
    t_interp = time.time() - t0

    print(f"      Interpreted {len(interpretations)} clusters")
    print(f"      Interpretation time: {t_interp:.1f}s")
    for interp in interpretations:
        status = "[OK]" if not interp.is_fallback else "[FALLBACK]"
        desc_preview = interp.description[:60].encode("ascii", "replace").decode()
        print(f"        Sense {interp.sense_id}: {status} - {desc_preview}...")
    print()

    # Save interpretations
    save_interpretations(target_word, interpretations)

    # ========================================================================
    # Step 5: Generate markdown report
    # ========================================================================
    print("[5/5] Generating demo report...")
    report_path = _generate_report(
        target_word=target_word,
        vectors_df=vectors_df,
        clustered_df=clustered_df,
        interpretations=interpretations,
        stats=stats,
        timings={
            "init": t_init,
            "encode": t_encode,
            "cluster": t_cluster,
            "interpret": t_interp,
            "total": t_init + t_encode + t_cluster + t_interp,
        },
        output_dir=output_dir,
        engine=engine,
    )

    print(f"      Report saved to: {report_path}")
    print(f"\n{'='*60}")
    print(f"  Demo complete! Total time: {t_init + t_encode + t_cluster + t_interp:.1f}s")
    print(f"{'='*60}\n")

    return report_path


def _generate_report(
    target_word: str,
    vectors_df: pd.DataFrame,
    clustered_df: pd.DataFrame,
    interpretations: List[SenseInterpretation],
    stats: dict,
    timings: dict,
    output_dir: Path,
    engine: AnalysisEngine,
) -> Path:
    """Generate a markdown demo report."""
    lines = []

    # Header
    lines.append(f"# PhilTrace Demo Report: \"{target_word}\"")
    lines.append("")
    lines.append(f"> Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> Encoder: `{engine.encoder.model_name_loaded}`")
    lines.append(f"> LLM: `{interpretations[0].model_used if interpretations else 'N/A'}`")
    lines.append("")

    # Summary
    lines.append("## 概要")
    lines.append("")
    lines.append(f"- 目标词: **{target_word}**")
    lines.append(f"- 语料句数: {len(vectors_df)}")
    lines.append(f"- 向量维度: {stats.get('shape', (0, 0))[1] if stats.get('shape') else 'N/A'}")
    lines.append(f"- 聚类数: {len(interpretations)}")
    labels = clustered_df["label"].values
    n_noise = int((labels == -1).sum())
    lines.append(f"- 噪声点: {n_noise}")
    lines.append(f"- 总耗时: {timings['total']:.1f}s")
    lines.append("")

    # Period distribution
    lines.append("## 时期分布")
    lines.append("")
    lines.append("| 时期 | 年代 | 句数 | 占比 |")
    lines.append("|------|------|------|------|")
    for period in PERIODS:
        count = len(vectors_df[vectors_df["period"] == period])
        pct = count / len(vectors_df) * 100 if len(vectors_df) > 0 else 0
        lines.append(f"| {period} | {PERIOD_YEARS[period]} | {count} | {pct:.1f}% |")
    lines.append("")

    # Cluster interpretations
    lines.append("## 语义簇诠释")
    lines.append("")

    for interp in interpretations:
        cluster_df = clustered_df[clustered_df["label"] == interp.sense_id]
        lines.append(f"### Sense {interp.sense_id} ({interp.period}, n={len(cluster_df)})")
        lines.append("")

        if interp.is_fallback:
            lines.append("> ⚠️ LLM 诠释失败，以下为检索回退结果")
            lines.append("")

        lines.append(f"**诠释**: {interp.description}")
        lines.append("")

        if interp.keywords:
            lines.append(f"**关键词**: {', '.join(interp.keywords)}")
            lines.append("")

        if interp.tradition:
            lines.append(f"**哲学传统**: {interp.tradition}")
            lines.append("")

        # Period breakdown for this cluster
        period_dist = cluster_df["period"].value_counts().to_dict()
        dist_str = " | ".join(f"{p}: {c}" for p, c in sorted(period_dist.items()))
        lines.append(f"**时期分布**: {dist_str}")
        lines.append("")

        # Representative samples
        lines.append("**代表样本**:")
        lines.append("")
        # Get top 3 samples closest to centroid
        if "vector" in cluster_df.columns and len(cluster_df) > 0:
            vecs = np.vstack(cluster_df["vector"].values)
            centroid = vecs.mean(axis=0)
            norms = np.linalg.norm(vecs, axis=1) + 1e-8
            sims = (vecs @ centroid) / (norms * (np.linalg.norm(centroid) + 1e-8))
            top_idx = np.argsort(sims)[-3:][::-1]
            for idx in top_idx:
                row = cluster_df.iloc[idx]
                lines.append(f"- > \"{row['text'][:200]}\"")
                lines.append(f"  > — *{row['source']}* ({row['period']})")
                lines.append("")
        lines.append("---")
        lines.append("")

    # Technical details
    lines.append("## 技术细节")
    lines.append("")
    lines.append("### 模型信息")
    lines.append("")
    model_info = engine.encoder.get_model_info()
    for k, v in model_info.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("### 算法参数")
    lines.append("")
    cfg = get_pipeline_config()
    lines.append(f"- UMAP: n_neighbors={cfg.umap.n_neighbors}, min_dist={cfg.umap.min_dist}, metric={cfg.umap.metric}")
    lines.append(f"- HDBSCAN: min_cluster_size={cfg.hdbscan.min_cluster_size}, min_samples={cfg.hdbscan.min_samples}")
    lines.append("")

    lines.append("### 耗时分解")
    lines.append("")
    lines.append(f"- 初始化: {timings['init']:.1f}s")
    lines.append(f"- 编码: {timings['encode']:.1f}s")
    lines.append(f"- 聚类: {timings['cluster']:.1f}s")
    lines.append(f"- LLM 诠释: {timings['interpret']:.1f}s")
    lines.append(f"- **总计: {timings['total']:.1f}s**")
    lines.append("")

    # Embedding statistics
    lines.append("### 嵌入统计")
    lines.append("")
    for k, v in stats.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.6f}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")

    # Write report
    report_path = output_dir / f"demo_{target_word}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="PhilTrace End-to-End Demo")
    parser.add_argument("--word", type=str, default="force", help="Target word to analyze")
    parser.add_argument("--provider", type=str, default="ollama", help="LLM provider (ollama/openai/lab_phil_llm)")
    parser.add_argument("--force", action="store_true", help="Force recompute (ignore cache)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()
    output_dir = Path(args.output) if args.output else None

    run_demo(
        target_word=args.word,
        provider=args.provider,
        force_recompute=args.force,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
