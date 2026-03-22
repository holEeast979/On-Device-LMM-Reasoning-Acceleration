# FasterOmni 实验数据位置索引

> 本文件记录本地 Results/ 目录与论文大纲表格的对应关系。
> 最后更新：2026-03-18

---

## 本地 Results/ 目录 → 论文大纲表格映射

| 目录 | 对应大纲表格 | 内容 |
|------|------------|------|
| `11_SchemeA_统一重跑_c8c9e7c/` | 表4-1, 表4-2, 表4-10 | 方案A主实验+编码器缓存+完整系统（已冻结） |
| `01_表4-5_表4-6_表4B-2_表4B-3_VideoMME_全量评估_6模式x300题/` | 表4-5, 表4-6, 表4B-2, 表4B-3 | Modality baselines + 音频贡献 |
| `02_表4B-4_Naive对比_kr0.5_Short108题/` | 表4B-4 | 帧选择策略对比（iframe/uniform/random） |
| `03_Naive对比_kr0.2_Short108题/` | — | kr=0.2 对比（补充参考） |
| `04_表4-3_Sparse64_OOM验证_Short108题/` | 表4-3 | Sparse@64 解决 OOM |
| `05_表4-4_表4B-1_Pareto_naive_iframe_kr扫描/` | 表4-4, 表4B-1 | Keep ratio 消融（kr=0.2~0.9） |
| `06_表4-7_表4B-5_MVBench_全量3600题/` | 表4-7, 表4B-5 | MVBench 极短视频退化 |
| `07_表4-8_表4B-6_VideoMME_ML边界实验/` | 表4-8, 表4B-6 | Video-MME M/L 失效分析 |
| `08_统计验证_Bootstrap_CI/` | 四（四）统计显著性验证 | Bootstrap CI 10000次重采样 |
| `09_Alpha_Kr_消融曲线/` | — | Alpha/Kr 消融可视化（补充参考） |
| `10_Adaptive_v2_smoke/` | — | Adaptive 探索（已放弃，历史记录） |

## 待补充

| 大纲表格 | 状态 | 说明 |
|---------|------|------|
| 表4-9 显存优化效果 | ⬜ 待实验 | 技术点4，峰值显存对比 |
| 表4-10 技术点3+4 行 | ⬜ 待实验 | RingBuffer + 显存优化 |
| 表4B-7 成本收益 | ✅ 可从现有数据计算 | 综合多个实验 |

## AutoDL 服务器原始数据

- SSH: `ssh -p 47964 root@connect.bjb1.seetacloud.com`
- 方案A统一重跑: `/root/autodl-tmp/results/fasteromni/scheme_a_c8c9e7c/`
- 历史实验: `/root/autodl-tmp/results/fasteromni/videomme_full/`
- MVBench: `/root/autodl-tmp/results/fasteromni/mvbench/`
- Pareto: `/root/autodl-tmp/results/fasteromni/pareto_naive_iframe/`
- Bootstrap: `/root/autodl-tmp/results/fasteromni/videomme_full/bootstrap_ci/`
