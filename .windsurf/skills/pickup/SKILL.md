---
name: pickup
description: 新对话开始时的接续流程。读取 PROGRESS.md 了解项目状态，快速进入工作状态。
---

# 新对话接续（/pickup）

当用户在新对话开始时说"接续"、"继续上次"、"读进度"时执行此流程。

> **最佳方式**：用户直接粘贴上次 `/handoff` 生成的交接 Prompt（更精准）。
> **兜底方式**：没有 Prompt 时用 `/pickup`，Agent 自己读 PROGRESS.md 理解状态。

## 步骤

1. **读取 PROGRESS.md**
   ```
   读取 /root/scripts/PROGRESS.md 全文（重点关注"当前状态"和"⬇️ 新对话 Agent 立即执行事项"）
   ```

2. **检查后台任务**
   ```bash
   # 检查是否有 tmux 实验在跑
   tmux list-sessions 2>/dev/null || echo "no sessions"
   # 如果有，查看进度
   tmux capture-pane -t eval -p | tail -20
   ```

3. **汇报当前状态**（BLUF 格式）
   - 一句话总结当前状态
   - 列出"立即执行事项"
   - 如果有后台实验，报告进度

4. **确认优先级**
   - 如果用户粘贴了交接 Prompt：按 Prompt 中的"立即执行"直接开始
   - 如果用户只说了 /pickup：问用户"按 PROGRESS.md 的计划继续，还是有新的优先级？"

5. **开始工作**
   - 按"⬇️ 新对话 Agent 立即执行事项"的第一条开始执行
   - 如果实验已完成，优先分析结果
   - 如果实验未完成，做其他待办事项
