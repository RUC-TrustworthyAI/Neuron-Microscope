# LLM 研究过程记录

[English](live-demo.md) · **简体中文**

**[神经元报告](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/)** · **[研究过程记录](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/walkthrough/?lang=zh)**

这份记录来自 **2026 年 10 月 29 日的 Qwen3-32B L55/N11703 研究**，使用 Diverse 100M 和 500 个采样位置。它保留了该次运行的解释、打标、实验、实验后修订和报告评阅。[当前可运行流程](workflow.md)会在实验前冻结解释，再根据实验结果分析该解释。保存的记录呈现其对应的操作与测量。

## 本地预览

从仓库根目录运行：

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory website
```

使用新版 Chrome、Edge、Firefox 或 Safari 打开 <http://127.0.0.1:8000/?lang=zh>。默认语言为英语，可以通过语言按钮或 `?lang=zh` 切换；切换时保留当前步骤。静态预览只需 Python 的 HTTP 服务与浏览器。

![Colorado 研究过程记录](../assets/pipeline-walkthrough-zh.jpg)

## 阅读证据

六个步骤依次展示证据准备、解释、样本打标、实验、修订，以及评阅和发布。每一步都链接到保存的任务、skill 指令、提示词节选、关键操作和对应产物。点击截图可以放大，保存的报告保留图表、悬停详情和折叠功能。

界面与讲解支持中英双语。提示词、样本上下文、测量和科学报告保留原文。最终报告绘制了共享 500 点样本中已标注的 495 个位置。

`website/` 与线上 walkthrough 展示同一个研究案例，包含整理后的节选与报告资源。完整 API 日志、凭证、权重和激活归档留在网站之外。报告资源经过压缩，由浏览器通过 `DecompressionStream` 加载。

## 将这份记录部署到 GitHub Pages

仓库具备 GitHub Pages 使用权限时：

1. 在 **Settings → Pages** 中选择 **GitHub Actions**。
2. 打开 **Actions → Publish live demo → Run workflow**，选择包含这些 `website/` 文件的分支。
3. 打开部署网址，通过 `?lang=en` 或 `?lang=zh` 选择语言。

这个工作流手动发布 `website/` 中的研究记录。神经元报告目录与视觉展示站各有对应的导出和托管入口。

## 导入准备好的公开记录

准备包含 `index.html`、JavaScript、CSS，以及 `content/`、`reports/`、`screenshots/` 的 walkthrough 目录，然后运行：

```bash
python3 examples/export_live_demo.py --source /path/to/prepared-walkthrough
```

命令将展示资产同步到 `website/`，保留报告和测量文件的原始字节；`--output` 可以指定其他目录。公开快照需要先选定提示词节选、截图和报告资源。导入步骤不调用 API 或执行模型测量。
