"""
矿井通风安全规程 RAG 批量测试脚本

覆盖 4 类查询场景：
  A. 数值参数查询    —— 验证 Parameter 节点 + vector 检索
  B. 设施规定查询    —— 验证 Facility 节点图索引 + hybrid 检索
  C. 多跳关系查询    —— 验证 GraphRAG 多跳遍历
  D. 合规辨识查询    —— 验证整体推理能力（最终目标场景）

用法：
    python ventilation_test_queries.py
    python ventilation_test_queries.py --out results.md
"""

import sys, os, time, argparse, logging
from dataclasses import dataclass
from typing import List

# 关闭 rag_modules 内部的 INFO 日志，只保留 pipeline 和测试级别
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logging.getLogger("ventilation_pipeline").setLevel(logging.INFO)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, PARENT_DIR)

from ventilation_rag_pipeline import VentilationRAGPipeline

# ══════════════════════════════════════════════════════════════
# 测试用例
# ══════════════════════════════════════════════════════════════
@dataclass
class TestCase:
    id:       str
    category: str     # A / B / C / D
    query:    str
    expect_hint: str  # 期望答案包含的关键词（用于判断是否命中）

TEST_CASES: List[TestCase] = [
    # ── A. 数值参数查询 ───────────────────────────────────────
    TestCase("A1", "数值参数",
        "岩巷掘进工作面最低风速要求是多少？",
        "0.15"),
    TestCase("A2", "数值参数",
        "采煤工作面的最高允许风速是多少？",
        "4"),
    TestCase("A3", "数值参数",
        "矿井内瓦斯浓度超过多少必须停止作业？",
        "1.0"),  # 答案格式为 "1.0 %"

    # ── B. 设施规定查询 ───────────────────────────────────────
    TestCase("B1", "设施规定",
        "局部通风机的安装位置有什么要求？",
        "局部通风机"),
    TestCase("B2", "设施规定",
        "风门的设置要求是什么？",
        "风门"),

    # ── C. 多跳关系查询 ───────────────────────────────────────
    TestCase("C1", "多跳关系",
        "主要通风机故障时应急处置程序是什么？",
        "备用"),
    TestCase("C2", "多跳关系",
        "回风巷与进风巷之间有哪些通风安全隔离要求？",
        "隔离"),

    # ── D. 合规辨识（最终目标场景）────────────────────────────
    TestCase("D1", "合规辨识",
        "某矿山采煤工作面测得风速为0.18m/s，且煤层倾角为5°，是否合规？如违规应如何整改？",
        "0.25"),
    TestCase("D2", "合规辨识",
        "工人在瓦斯浓度1.2%的环境下继续作业，是否违规？依据哪条规程？",
        "违规"),  # 期望系统能通过 fallback 找到瓦斯浓度相关条款
]

# ══════════════════════════════════════════════════════════════
# 评估结果记录
# ══════════════════════════════════════════════════════════════
@dataclass
class TestResult:
    case:        TestCase
    strategy:    str    # hybrid_traditional / graph_rag / combined
    doc_count:   int
    elapsed_s:   float
    answer:      str
    hit:         bool   # 是否包含期望关键词

# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════
def run_tests(top_k: int = 5) -> List[TestResult]:
    print("=" * 65)
    print("  矿井通风安全规程 RAG 批量测试")
    print("=" * 65)

    pipeline = VentilationRAGPipeline()
    pipeline.initialize()
    print(f"\n  初始化完成，开始运行 {len(TEST_CASES)} 条测试\n")
    print("-" * 65)

    results = []
    for tc in TEST_CASES:
        print(f"\n[{tc.id}] {tc.category} | {tc.query}")

        t0 = time.time()
        try:
            # 调用统一入口，确保 fallback 逻辑生效
            answer   = pipeline.query(tc.query, top_k=top_k)
            strategy = getattr(pipeline, '_last_strategy', 'unknown')
            doc_count= getattr(pipeline, '_last_doc_count', -1)
        except Exception as e:
            strategy  = "ERROR"
            doc_count = 0
            answer    = f"[ERROR] {e}"

        elapsed = time.time() - t0
        hit     = tc.expect_hint in answer

        indicator = "✓" if hit else "✗"
        print(f"  策略: {strategy:<22} 文档数: {doc_count}  耗时: {elapsed:.1f}s  命中: {indicator}")
        print(f"  答案节选: {answer[:120].replace(chr(10), ' ')}…")

        results.append(TestResult(tc, strategy, doc_count, elapsed, answer, hit))

    pipeline.close()
    return results


def write_report(results: List[TestResult], path: str):
    lines = ["# RAG 批量测试报告\n"]

    # 统计
    total    = len(results)
    hit_cnt  = sum(1 for r in results if r.hit)
    trad_cnt = sum(1 for r in results if r.strategy == "hybrid_traditional")
    graph_cnt= sum(1 for r in results if r.strategy == "graph_rag")
    comb_cnt = sum(1 for r in results if r.strategy == "combined")
    avg_t    = sum(r.elapsed_s for r in results) / total

    lines += [
        "## 概览统计\n",
        f"| 指标 | 值 |",
        f"|------|----|",
        f"| 总测试数 | {total} |",
        f"| 关键词命中率 | {hit_cnt}/{total} ({hit_cnt/total*100:.0f}%) |",
        f"| 混合传统检索 | {trad_cnt} 次 |",
        f"| 图RAG检索   | {graph_cnt} 次 |",
        f"| 组合检索    | {comb_cnt} 次 |",
        f"| 平均耗时    | {avg_t:.1f}s |",
        "",
    ]

    # 按类别分组
    categories = sorted(set(r.case.category for r in results))
    for cat in categories:
        cat_results = [r for r in results if r.case.category == cat]
        lines.append(f"\n## {cat}\n")
        for r in cat_results:
            hit_mark = "✅" if r.hit else "❌"
            lines += [
                f"### [{r.case.id}] {r.case.query}",
                f"- 路由策略: `{r.strategy}` | 文档数: {r.doc_count} | 耗时: {r.elapsed_s:.1f}s | 命中: {hit_mark}",
                f"- 期望关键词: `{r.case.expect_hint}`",
                "",
                "<details><summary>完整答案</summary>",
                "",
                r.answer,
                "",
                "</details>",
                "",
            ]

    # 问题分析
    failed = [r for r in results if not r.hit]
    if failed:
        lines.append("\n## 未命中分析\n")
        for r in failed:
            lines += [
                f"- **[{r.case.id}]** `{r.case.query}`",
                f"  - 策略: {r.strategy}，文档数: {r.doc_count}",
                f"  - 期望: `{r.case.expect_hint}`",
                f"  - 答案节选: {r.answer[:200]}",
                "",
            ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n\n报告已保存到：{path}")


def print_summary(results: List[TestResult]):
    print("\n" + "=" * 65)
    print("  测试汇总")
    print("=" * 65)
    total   = len(results)
    hit_cnt = sum(1 for r in results if r.hit)
    print(f"  关键词命中率: {hit_cnt}/{total} ({hit_cnt/total*100:.0f}%)")
    print(f"  路由分布：")
    for strategy in ("hybrid_traditional", "graph_rag", "combined", "ERROR"):
        cnt = sum(1 for r in results if r.strategy == strategy)
        if cnt:
            bar = "█" * cnt
            print(f"    {strategy:<25} {bar} ({cnt})")
    print(f"  平均耗时: {sum(r.elapsed_s for r in results)/total:.1f}s")

    failed = [r for r in results if not r.hit]
    if failed:
        print(f"\n  未命中 ({len(failed)} 条)：")
        for r in failed:
            print(f"    [{r.case.id}] {r.case.query[:45]}…")
            print(f"         期望: '{r.case.expect_hint}' | 策略: {r.strategy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量测试通风 RAG 系统")
    parser.add_argument("--out", default="test_results.md",
                        help="结果报告输出路径（默认 test_results.md）")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results = run_tests(top_k=args.top_k)
    print_summary(results)

    out_path = os.path.join(BASE_DIR, args.out)
    write_report(results, out_path)
