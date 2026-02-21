# Jarvis GitHub 文档拉取指南

## 项目文档结构

FasterOmni 项目有两个核心进度文档：

### 1. PROGRESS.md（日常监控用）
- **用途**：当前状态快照、待做工作、数据保护索引
- **长度**：~200 行（精简版）
- **更新频率**：每次实验完成后
- **GitHub URL**：
```
https://raw.githubusercontent.com/holEeast979/On-Device-LMM-Reasoning-Acceleration/master/PROGRESS.md
```

### 2. PROGRESS_ARCHIVE.md（深度查询用）
- **用途**：完整历史记录、实验数据详情、变更日志
- **长度**：~728 行（完整版）
- **更新频率**：归档时
- **GitHub URL**：
```
https://raw.githubusercontent.com/holEeast979/On-Device-LMM-Reasoning-Acceleration/master/PROGRESS_ARCHIVE.md
```

## 使用场景

### 日常监控（只拉 PROGRESS.md）
- 查看当前实验进度
- 了解待做工作
- 检查数据保护规则

### 深度查询（需要拉 PROGRESS_ARCHIVE.md）
- 查找某个实验的完整数据（如"Phase 1 kr 消融的具体数值"）
- 了解某个决策的历史背景（如"为什么 Medium 音频干扰"）
- Troubleshooting 时查找根因（如"[2.20] 代码污染事件详情"）

## 拉取命令示例

### 方法 1：直接读取 GitHub raw URL
```bash
# 拉取精简版
curl -s https://raw.githubusercontent.com/holEeast979/On-Device-LMM-Reasoning-Acceleration/master/PROGRESS.md

# 拉取完整归档
curl -s https://raw.githubusercontent.com/holEeast979/On-Device-LMM-Reasoning-Acceleration/master/PROGRESS_ARCHIVE.md
```

### 方法 2：在对话中请求
如果你的 Agent 支持 GitHub 集成，可以直接说：
- "拉取 PROGRESS.md"
- "查看 PROGRESS_ARCHIVE.md 中关于 Modality baselines 的数据"

## 记忆建议

将以下内容加入你的记忆：

**标题**：FasterOmni 项目文档 URL

**内容**：
```
FasterOmni 项目有两个核心文档：

1. PROGRESS.md（日常用，~200行）
   URL: https://raw.githubusercontent.com/holEeast979/On-Device-LMM-Reasoning-Acceleration/master/PROGRESS.md
   用途：当前状态、待做工作、数据保护索引

2. PROGRESS_ARCHIVE.md（深度查询用，~728行）
   URL: https://raw.githubusercontent.com/holEeast979/On-Device-LMM-Reasoning-Acceleration/master/PROGRESS_ARCHIVE.md
   用途：完整历史、实验数据详情、变更日志

日常监控只需拉取 PROGRESS.md。
需要历史细节或完整数据时，拉取 PROGRESS_ARCHIVE.md。
```

**标签**：`fasteromni`, `github`, `documentation`, `progress_tracking`
