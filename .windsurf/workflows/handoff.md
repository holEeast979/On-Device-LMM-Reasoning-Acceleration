---
description: 对话结束时的交接流程。更新 PROGRESS.md 并生成新对话的 Prompt，确保下一个 Agent 能无缝接续。
---

# 对话结束交接（/handoff）

当用户说"准备交接"、"结束对话"、"更新进度"时执行此流程。

## 步骤

1. **读取当前 PROGRESS.md**
   ```
   读取 /root/scripts/PROGRESS.md 的"当前状态"和"待做工作"部分
   ```

2. **更新 PROGRESS.md 的"当前状态"部分**
   - 更新日期时间标记（如 `2.23 凌晨`）
   - 更新一句话状态摘要（加粗）
   - "✅ 已完成"列表：补充本次对话新完成的工作
   - "🔄 正在进行"列表：记录后台任务（如 tmux 实验）
   - "⬇️ 新对话 Agent 立即执行事项"：写出具体的、可执行的步骤（包括命令），按优先级排序

3. **更新 PROGRESS.md 的"待做工作"表格**
   - 已完成的任务标 ✅
   - 新增的待做任务加入表格
   - 正在跑的实验标 🔄 并注明输出目录

4. **更新变更日志**
   - 在"变更日志"最上方添加本次对话的条目
   - 格式：`- **[日期时间]** 简要描述。关键 commit hash`

5. **Commit + Push**
   ```bash
   cd /root/scripts && git add PROGRESS.md && git commit -m "docs: handoff update - [一句话描述]" && git push origin master
   ```

6. **生成新对话 Prompt**
   
   输出一段 Prompt（用代码块包裹），用户直接复制粘贴到新对话即可：
   
   ```
   请先读取 /root/scripts/PROGRESS.md，了解项目当前状态。
   
   ## 上次对话交接摘要
   [一句话：上次做了什么]
   
   ## 当前状态
   [从 PROGRESS.md 的"当前状态"部分提取关键信息]
   
   ## 立即执行
   [从 PROGRESS.md 的"⬇️ 新对话 Agent 立即执行事项"提取]
   ```

7. **调用检查点工具**确认用户收到 Prompt
