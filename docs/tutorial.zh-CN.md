# 从安装到原生评分：中文教程

[English](tutorial.md) · [中文 README](../README.zh-CN.md) · [参数参考](configuration.md)

按顺序区分三个输出：**修改后的输入 → 模型生成的答案 → 原生任务评分**。建议使用新工作目录和 Python 3.10+；密钥始终留在环境变量或安全凭据管理器中。

## 1. 安装与零调用入门

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install agentinputlab==0.1.0a20
agentinputlab --version
agentinputlab doctor --offline --json
agentinputlab smoke --json
agentinputlab config init > run.json
agentinputlab config validate run.json
agentinputlab config run run.json --query "Find a suitable flight"
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活。最后一步运行 `plan.plan_and_solve.ps_prompt`，可在 `state.instruction` 中看到新指令。这不是航班答案，也没有调用模型。`validate` 检查配置，`doctor` 检查本机依赖，不代表远程模型和原生环境已就绪。

对应 Python 调用：

```python
from agentinputlab import RunConfig
config = RunConfig.from_file("run.json")
result = config.run("Find a suitable flight")
print(result["state"]["instruction"])
```

更换方法前，先查看精确的 profile 和参数：

```sh
agentinputlab list --releases --search rrr
agentinputlab show rewrite.rrr.common_support_prompt --parameters
agentinputlab datasets list --search longbench
agentinputlab datasets show longbench.multifieldqa_en
```

## 2. 接入真实 API：改写、检索、回答

从公开仓库取得 [examples/rrr_api.py](../examples/rrr_api.py)。安装 PyPI 包后即可使用该脚本，不需要研究仓库。通过安全终端或凭据管理器配置 `AIL_API_KEY`，不要把密钥写入 JSON 或提交 Git。以下三个非敏感变量必须改成你实际服务的值：

```sh
export AIL_BASE_URL="https://your-provider.example/v1"
export AIL_MODEL="your-served-model-id"
export AIL_MODEL_REVISION="your-deployment-revision"
python examples/rrr_api.py > rrr.json
agentinputlab config validate rrr.json
agentinputlab config run rrr.json --query "What is the capital of France?"
```

生成配置不消耗模型调用；最后一个命令才会真实执行。该示例使用 RRR 提示词适配版：改写问题→从小型 BM25 语料检索一条文档→读取并回答。允许两次模型调用、一次检索，每次最多 512 output tokens，累计最多 1024。查看回答、usage 和执行状态；这不是官方基准结果，也不是训练后 PPO 权重的复现。

也可用 `python examples/rrr_api.py --execute` 直接执行同一示例。若服务返回不同的模型别名，或需要显式 thinking／结构化输出设置，在 `backend.response_model_aliases`／`backend.extra_body` 中按服务实际能力配置，见[参数参考](configuration.md)。`temperature` 等请求自身的字段不要塞进 `extra_body`。

该 profile 返回输入 patch，reader 的答案保存在 `result["record"]["patch"]["trace"]["events"]` 中名为 `frozen_reader_output` 的事件的 `payload.answer`。改写后的问题位于 `result["state"]["query"]`，用量位于 `result["record"]["usage"]`。`unpriced_model_calls > 0` 表示没有获得货币费用，不代表 API 免费。

### 用户如何自由配置

| 需求 | `rrr.json` 中的位置 |
| --- | --- |
| 累计输出 tokens／模型调用次数 | `budget.max_output_tokens`／`budget.max_model_calls` |
| 每个阶段的输出上限 | `parameters.config.max_rewriter_tokens`／`max_reader_tokens` |
| temperature | `parameters.config.temperature` |
| 检索数量／语料 | `parameters.config.top_k`／`retriever.documents` |
| BM25 参数 | `retriever.k1`／`retriever.b` |
| 模型与 API 地址 | `backend.model_id`／`backend.base_url` |

其他方法未必用相同字段，先看 `show ... --parameters`。`max_length` 可能表示输入加输出长度或压缩长度，不是通用生成参数。只改总预算不会自动提高每阶段上限；拼错的参数会报错。

## 3. 跑真实数据集并取得原生评分

以 LongBench MultiFieldQA-en 的一条固定顺序样本为例：

```sh
python -m pip install 'agentinputlab[longbench]==0.1.0a20'
```

按[数据准备指南](benchmark-setup.md)取得固定版本的数据压缩包、原生评分器和提示词模板，确认其许可。安装包不附带这些第三方资源。已有相同哈希的文件直接复用，不必再次下载。创建 `longbench.json`：

```json
{"data_zip":"/path/to/data.zip","scorer_path":"/path/to/metrics.py","prompts_path":"/path/to/dataset2prompt.json"}
```

在 Python 中生成模型配置与评分执行配置：

```python
import json, os
config = {
    "release_id": "baseline.no_op", "seed": 17,
    "budget": {"max_output_tokens": 512, "max_model_calls": 1},
    "backend": {
        "model_id": os.environ["AIL_MODEL"],
        "base_url": os.environ["AIL_BASE_URL"],
        "deployment_identity": {"revision": os.environ["AIL_MODEL_REVISION"]},
        "api_key_env": "AIL_API_KEY"
    }
}
with open("native.json", "x", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
with open("predictor.json", "x", encoding="utf-8") as f:
    json.dump({"schema_version": 1, "native_text": {
        "max_output_tokens": 512, "temperature": 0.0}}, f)
```

这里 `baseline.no_op` 仅表示不修改输入，评分流程仍会调用下游模型；它不自动等于论文中的 ReAct 配置。

```sh
agentinputlab config evaluate native.json \
  --benchmark longbench.multifieldqa_en --split test --limit 1 \
  --longbench-config longbench.json --predictor-config predictor.json \
  --output native-evaluation.jsonl
```

输出记录每条样本的开始与终态，完成后包括预测、原生指标和用量。检查 `scored_count`、失败记录和原始分母；低分是模型表现，不等于安装失败。一条样本通过不能当作整个榜单成绩。要比较 Plan-and-Solve，将配置另存，改 `release_id` 为 `plan.plan_and_solve.ps_prompt`，使用新的输出路径；不要覆盖旧结果。

第一次使用且没有本地资源时，阅读上游许可后，可将 `--longbench-config longbench.json` 替换为 `--cache-dir ./longbench-cache --accept-license-review`，自动下载并校验固定版本。六个子集共享约 114 MB 的原始下载文件；后续复用缓存。

Python 对应入口是 `RunConfig.evaluate(..., longbench_sources=..., predictor_parameters=...)`，详见[配置文档](configuration.md)。

## 4. 工作流与网页操作

先用无需模型的例子体验准备、执行和状态保存：

```sh
agentinputlab workflow init --query "Find a suitable flight" > workflow.json
agentinputlab workflow inspect workflow.json
agentinputlab workflow prepare workflow.json --output-dir prepared-workflow
agentinputlab workflow run prepared-workflow/workflow.json --execute --output-dir workflow-run
agentinputlab workflow status workflow-run
agentinputlab-studio --workflow-root ./studio-workflows
```

打开打印出的 `/workflows` 地址。选择预设、方法和模型，编辑预算及高级 JSON，生成配置、检查前置条件后再执行。网页与命令行共用执行引擎；缺少 checkpoint、训练产物或原生环境时，需要先补齐这些资源。下载授权、并发和断点续接见[工作流指南](workflows.md)。有保存记录时优先恢复工作流，不要换个输出路径盲目重跑。

服务器使用 SSH 隧道访问，Studio 保持绑定 localhost；它是受信任操作者界面，不是公网上的多人服务。网页中只填密钥环境变量名称，不填密钥值，截图也不要包含密钥。

## 5. 常见问题和验收

| 现象 | 处理 |
| --- | --- |
| 找不到命令 | 激活虚拟环境，或用 `python -m agentinputlab` |
| GitHub 下载超时 | 使用官方 PyPI，不要关闭 TLS 校验 |
| 缺密钥／401／403 | 检查变量名称、账户与服务权限 |
| 429／额度耗尽 | 等待额度或调整并发，不要盲目重复已消耗请求 |
| 缺 runtime／artifact | 查所选 profile 的参数和对应准备指南 |
| 返回模型身份或 schema 不符 | 核对 provider 的实际身份和传输模式 |
| 配置通过但没有回答 | 检查是否选择了只生成输入 patch 的方法 |
| 输出文件已存在 | 查看原记录；支持的工作流用恢复入口，保留旧尝试 |

验收分为包安装、接口检查、旧响应回放、真实模型调用和原生评分。参见[验收标准](validation.md)。保留版本、配置、资源哈希、样本 ID 和终态，才能区分“这个组合实际跑过”和“接口可以选择”。
