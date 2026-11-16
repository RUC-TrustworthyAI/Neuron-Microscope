# Neuron Microscope 本地协作指南

## 工作位置

- 在 `repo_fork/Neuron-Microscope` 修改和验证。`repo_origin/Neuron-Microscope` 用于核对上游，验证通过后再按用户安排合并。开始前先确认当前目录与 Git 状态，保留已有改动。
- 每组实验使用独立的 `runs/<run>` 目录；临时脚本可放 `.local/`。相同配置可以续跑，更换模型、语料、神经元或测量设置时使用新目录。模型、数据、凭证、生成产物和本机配置不要提交到上游。
- 视觉子模块另有 [`vision/AGENTS.md`](vision/AGENTS.md)；在 `vision/` 工作时也读取该文件。

## 本机环境

- 查看回放和开发报告页面可以在 Windows 上完成。语言侧采集及完整测量使用 WSL 2 的 `Ubuntu` 和 NVIDIA CUDA：原生 Windows 缺少 `fcntl`，常驻 GPU worker 还需要 Linux 的 `/proc`。
- 复用 Miniforge 的 Conda 环境 `llm-cuda`。在 WSL 中用 `conda activate llm-cuda`，非交互执行可用 `~/miniforge3/bin/conda run -n llm-cuda --no-capture-output ...`。环境与现有前向检查见 [WSL 指南](../../task/wsl-cuda/README.md)。
- RTX 4060 Laptop 有约 8 GiB 显存。本地 dense Qwen3 权重在 `~/.local/share/llm-cuda/models/`，不在默认 Hugging Face 缓存下。`Qwen3-0.6B` 已用 BF16 跑通采集、报告、RelP 和消融；`Qwen3-1.7B` 已下载，WSL 指南有其 BF16 前向记录，可用于更贴近真实模型的本机验证。前向可用不代表长上下文归因或大批量可用；先用 `--gpus 0`、短前缀和小批量。复现原报告的 Qwen3-32B 需更大显存的 Linux 主机。
- 优先复用本地权重和现有依赖。API 直接连接；Hugging Face 用 `HF_ENDPOINT=https://hf-mirror.com`，Conda 用清华镜像。不要为例行验证额外跑整文件 SHA-256；保留现有工具用于运行、续跑或版本约束的检查。

## 语言侧工具与验证

- `python -m language.collection` 从本地 JSONL（每行含 `text`）或 Parquet 采集原生 `gate`、`up`（归档字段 `value`）和 `product`，生成归档与 `reports/<neuron>/index.html`。最小集成验证可用约 1,024 个 token、一个有效的 `Lx_Ny` 坐标、短前缀和一个报告 worker。参见[数据指南](language/docs/data.md)。
- `language/examples/configure.py` 用本地模型、激活归档和初始报告生成工作流配置，初始报告需含 `effects.json`。`python -m language` 执行完整调查；`--stop-after interpretation` 不加载目标模型，但仍调用解释 API。只检查报告渲染时，使用 `language/examples/demo.py` 或直接读 `website/` 中的已有回放。
- 先验证产物，再扩大规模：检查采集 `status.json`、原生三通道和 token 数量、报告及其结构化数据；改动归因或干预时，再测一个实际位置，并核对 CUDA 设备、干预记录和显存峰值。区分前向检查、初始报告和完整解释流程，避免把小规模验证当作 32B 的科学复现。
- 保留实际模型、精度、输入前缀、评分位置和种子。解释修订用过的样本不能再当独立测试集；RelP 是归因，因果效应用原生干预测量；Blind／Search 的激活极值分数不等于机制正确性。
- 正式解释复用原仓库的完整采样、初始解释和两次修订、三次盲标与选择逻辑，冻结选中解释及其自身预测。局部快速检查不能替代这一步。追加实验从上游报告分析的 Criticism 出发，保留原条目和冻结解释，按 Colorado 公开案例的方法最多两轮，每轮最多五项；第二轮同时读取最初的 Criticism 和第一轮全部结果。没有新的可区分问题时提前停止，不为凑齐 5+5 项生成实验。报告采用原版紧凑 Criticism 行，已有证据读取、局部修订和完整测量保存在运行档案，不重复展示。

## 原版参考资料

- 工作区的 [Already Clean 参考库](../../references/already-clean/README.md) 包含 Goodfire 原文、官方 PDF 和原版 Colorado 报告。用原版核对方法和展示，独立解释和盲测时保持隔离；需要生成双栏对照时，在导出阶段将该目录作为 `references` 输入。
