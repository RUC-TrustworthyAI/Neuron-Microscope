# Neuron Investigator

一个研究 dense Qwen3 原生 MLP neuron 的三阶段 agent pipeline：从已有激活提出解释，设计并测量实验，再根据测量修订解释，生成可交互 HTML。

[English](README.md) · [安装与运行](docs/running.md) · [输入格式](docs/data.md) · [方法与代码](docs/method.md)

仓库只包含代码、三个 skills、软件测试及不含数据的页面模板。模型、激活、实验结果、原博客页面快照、API 日志和密钥均不上传。

## 流程

1. **准备证据**：从已有激活按幅度区间抽候选点，各区间最多 40 点，再按候选数量比例保留最多 256 点，交给 DeepSeek 标注语义。此步骤在初步解释之前。
2. **初步解释**：第一个独立容器只装 interpret skill 和目标 neuron 的 HTML、结构化证据。
3. **实验**：第二个独立容器装 experiment skill；agent 设计实验，固定工具在宿主机完成替换、搜索、消融、RelP 归因及其他测量。
4. **修订**：第三个独立容器装 revise skill，根据测量修订解释、Criticism 和图注；固定代码生成 HTML 与费用章节。
5. **发布**：若提供对应原博客的本地快照，发布工具自动加入左右分屏、版块对齐和同步滚动。官方内容只加入展示页，不进入研究 agent 的输入。

Clusters 使用 agent 明确指定的语义成员，可跨 route 合并或拆分；原始 NMF routes 保留。弱激活、反例和未归类成员不会因为不符合名字就被删除。数字、占比及例句来自原始测量。

## 安装

完整流程需要 Linux、Bash 4+、Docker、Python 3.10+、本地 dense-Qwen3 权重、已有激活和初始报告包，以及 DeepSeek API 凭证。默认 DeepSeek high。GPU 只承担目标模型的测量，四卡共同承载一份模型。

```bash
git clone https://github.com/RUC-TrustworthyAI/Neuron-Investigator.git
cd Neuron-Investigator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/gpu.txt
# 按驱动选择 CUDA wheel；以下为研究环境版本。
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
docker build -t neuron-investigator:local .
cp settings.json settings.local.json
```

修改 `settings.local.json` 中的 `paths.reports`、`paths.activations`、`paths.model`、`paths.state` 和 `gpu_groups`。可选 `paths.references` 放 `Lx_Ny.html` 原博客快照。相对路径以配置文件所在目录为基准。报告包及激活格式见 [docs/data.md](docs/data.md)。

在宿主机环境中提供 `DEEPSEEK_API_KEY`，或通过 `NEURON_PAPER_API_KEY_FILE` 指向本地密钥文件。不要把密钥写进配置或 Git。

## 运行与查看

```bash
NEURON_PAPER_BUDGET_USD=5 ./run.sh prepare \
  --settings settings.local.json --output runs/example --neurons L55_N11703
./run.sh execute --run runs/example
./run.sh publish --run runs/example --neuron L55_N11703 --output results/example
python -m http.server 8000 --directory results/example
```

打开 **http://localhost:8000/**。示例 neuron 需替换为自己的数据中存在的编号。

- `prepare`：冻结代码和配置，不启动 LLM 或模型实验。
- `execute`：按快照执行准备和三阶段流程。
- `publish`：保留生成的科学内容，加入统一显示与可选对比功能。
- 默认整批 API 上限 2 美元；示例设为 5 美元。设 `unlimited` 可取消上限。费用是配置单价计算的估算，不是账户账单。
- `--neurons` 后可列多个 neuron。默认前一个第二阶段结束后才启动下一个，第三阶段可与下一个第一阶段重叠。
- 完整请求、输出、测量、费用和耗时保存在 `runs/`。旧结果保留；这些目录被 Git 忽略。

更多续跑、单阶段和复用测量操作见 [运行说明](docs/running.md)。页面中 Exemplars 与 Clusters 只截短展示上文，计算仍使用采集时完整前缀和明确评分位置。

## 无 GPU 演示

```bash
python -m pip install -r requirements/cpu.txt
python examples/demo.py --output runs/demo
python -m http.server 8000 --directory runs/demo
```

演示使用代码生成的合成数据，仅检查页面功能，不是实验结果。它不需要 API、Docker 或模型，也可以在 macOS 上运行。

代码结构、方法来源和第三方说明见 [英文 README](README.md)、[release.md](docs/release.md) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
