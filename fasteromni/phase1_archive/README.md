# Phase 1 归档文件

这些文件是 Phase 1 早期开发的独立脚本，在切换到 Video-MME 选择题评估后已被 `eval_videomme.py` 取代。

| 文件 | 用途 | 取代者 |
|------|------|--------|
| `run_ablation.py` | 消融实验（扫描 kr/alpha） | `eval_videomme.py --sweep` |
| `run_comparison.py` | Baseline vs Sparse 对比 | `eval_videomme.py --modes baseline sparse` |
| `eval_accuracy.py` | ActivityNet-QA 准确率评估 | `eval_videomme.py` |
| `evaluator.py` | 规范化 EM 评估器（VQA 标准） | Video-MME 选择题不需要 |
| `analyze_scoring.py` | AV-LRM 打分分布分析 | 调试工具，偶尔使用 |
| `analyze_gop.py` | GOP 结构统计分析 | 调试工具，偶尔使用 |

归档时间：[2.20]
