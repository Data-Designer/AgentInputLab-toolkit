# AgentInputLab

[English](README.md) · **简体中文**

面向 LLM 智能体的输入干预工具包：通过 Python、命令行、可编辑 JSON 和本地 Web 界面，把方法、模型、检索器与数据集执行／评分连接起来。

[网站](https://data-designer.github.io/AgentInputLab/) · [PyPI](https://pypi.org/project/agentinputlab/) · [公开源码](https://github.com/Data-Designer/AgentInputLab-toolkit) · [完整中文教程](docs/tutorial.zh-CN.md)

公开仓库包含库代码、适配器、合成测试和使用文档，不包含研究数据、实验日志、模型权重或私人配置。JavaScript SDK 目前通过源码构建使用，尚未发布到 npm。

## 1. 安装和第一次运行

需要 Python 3.10+；可选模型依赖有各自的 Python／平台要求。建议使用独立虚拟环境。

```sh
python -m venv .venv
# Linux / macOS；Windows PowerShell 使用 .venv\Scripts\Activate.ps1
source .venv/bin/activate
python -m pip install agentinputlab==0.1.0a19
python -m agentinputlab --version
python -m agentinputlab doctor --offline --json
python -m agentinputlab smoke --json
agentinputlab config init > run.json
agentinputlab config validate run.json
agentinputlab config run run.json --query "Find a suitable flight"
```

这里运行的 Plan-and-Solve 配置会改变输入指令，**不调用模型，不返回航班答案，也不产生任务评分**。真实模型调用与数据集评分见[教程](docs/tutorial.zh-CN.md)。`doctor` 和 `smoke` 检查本地环境，不代表远程模型已连通。

也可以从公开源码安装：

```sh
git clone https://github.com/Data-Designer/AgentInputLab-toolkit.git
cd AgentInputLab-toolkit
python -m pip install .
```

若服务器无法访问 GitHub，可直接使用 PyPI；不要通过关闭 TLS 校验解决下载问题。

## 2. 支持哪些操作

| 目标 | 入口 | 需要准备什么 |
| --- | --- | --- |
| 发现方法和参数 | `list --releases`、`show RELEASE --parameters` | 无模型或数据下载 |
| 查询数据集接口 | `datasets list`、`datasets show ID` | 查看原生运行时、数据来源和准备要求 |
| 自定义运行 | `config init / validate / run`、Python `RunConfig` | 方法参数、预算、模型／检索器配置 |
| 准备→执行→评分 | `workflow inspect / prepare / run / status` | 所选数据集及方法的真实依赖 |
| 浏览器操作 | `agentinputlab-studio` | 本地回环地址；与命令行共用配置和执行逻辑 |
| 接入自己的方法／模型 | [执行接口](docs/real-execution.md) | 遵守状态、预算、返回值和评分接口 |

五类输入方法覆盖改写、分解、获取信息、澄清和上下文整理。每个方法可能提供多个 release profile；训练版、原生版、受控适配版并不是可任意互换的同一个配置。查看具体 profile 的参数和依赖后再运行。

## 3. 配置、模型与数据

```sh
agentinputlab list --releases --search rrr
agentinputlab show rewrite.rrr.common_support_prompt --parameters
agentinputlab datasets list --search longbench
agentinputlab datasets show longbench.multifieldqa_en
```

可配置累计 tokens／调用次数、每次生成长度、temperature、seed、BM25 参数和外部检索器。参数归属及名称随方法变化；`max_length` 不是所有方法通用的输出 token 参数。未知字段会报错，不会悄悄忽略。

API 密钥使用环境变量，不写入 JSON、不提交 GitHub。支持 OpenAI-compatible 服务，但模型身份、结构化输出、工具调用等能力需要按实际 provider 配置。仅填写模型名不能替代本地权重或外部运行时。

## 4. 文档导航

- [中文教程：离线入门、真实 API、原生评分和 Web](docs/tutorial.zh-CN.md)
- [英文教程](docs/tutorial.md)
- [参数与检索器配置](docs/configuration.md)
- [数据集准备](docs/benchmark-setup.md)
- [工作流与恢复](docs/workflows.md)
- [Web Studio](docs/studio-workflows.md)
- [安装包／公开源码验收标准](docs/validation.md)
- [故障排查](docs/troubleshooting.md)

安装后可离线读取指南：`agentinputlab docs --list` 和 `agentinputlab docs configuration`。从 a16 起，中英文教程也随包提供：`agentinputlab docs tutorial`、`agentinputlab docs tutorial.zh-CN`。

## 5. 结果应该怎样理解

方法执行、模型回答和原生任务评分是不同阶段。评分依赖数据集的原始评分器；执行失败和缺分会保留，不会自动伪造成成功或得分。相同请求的旧响应回放用于验证接线，不算新的模型调用。查看[分层验收说明](docs/toolkit-acceptance.md)，不要把接口可发现等同于任意方法×数据集×模型都已实测。

## 贡献与引用

欢迎提交可复现问题：附上版本、profile、脱敏配置、错误类型与最小操作步骤；不要上传密钥、私人数据或完整研究日志。请先阅读 [CONTRIBUTING](CONTRIBUTING.md)、[许可](LICENSE) 和 [第三方声明](THIRD_PARTY_NOTICES.md)。引用信息见 [CITATION.cff](CITATION.cff)。
