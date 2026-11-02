# 参数化电商 RSI 训练与本机实时看板

入口：`sh scripts/ecom-train.sh`，或 `.venv/bin/python -m harness2.cli.ecom_train`。
在源码 checkout 内运行，需要现有 `.venv`、MerchantBench/OpenCode 和
[`ecom-clean.sh` 的模型配置](CLEAN_ENVIRONMENT.md)。新增工具仅用 Python 标准库和原生 HTML/CSS/JS。

**`run`、`resume` 会调用模型并产生费用。`--dry-run`、`serve`、`status` 不调用模型。**
本工具是持续学习训练入口，不是冻结的配对评测；独立泛化评估请用
[冻结配对评测工具](ECOM_PAIRED.md)，并使用未参与训练的 seed。

## 预览与启动

```bash
# 先预览：seed 0–9，每任务 24 个模拟小时；不创建文件，不启动服务，不写经验。
sh scripts/ecom-train.sh run --tasks 10 --seed-start 0 --hours 24 --dry-run

# 确认预算后执行。会创建全新实验目录并启动脱离终端的 worker 和只读看板。
sh scripts/ecom-train.sh run --tasks 10 --seed-start 0 --hours 24 \
  --k 1 --mode parallel --order-seed 0 --timeout 1800 --port 18766
```

命令打印实验目录、worker PID、看板 URL、日志位置。关闭启动终端不停止训练。
默认实验目录为 `runs/ecom-train-<时间>-<随机后缀>`；也可指定 `--output runs/my-training`，
**指定目录必须尚不存在**。训练先预检，然后串行执行 synthetic domain 的任务，
共用本批实验的经验库。每个任务按原 RSI 算法做改进、经营探测、合成及经验写回。

任务顺序使用核心 `stream_order`，不是简单 seed 升序。例如 `--order-seed 0`
控制整个任务流的确定性打乱，不会修改 `master_seed`。
`--workers 1 --split all --max-tasks <任务数>` 由工具固定，避免同一 domain 内并行写经验。

### 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--tasks` | `10` | synthetic 任务数量，正整数，上限 1000；更大实验需显式拆批 |
| `--seed-start` | `0` | 连续任务 seed 的起点，非负整数 |
| `--seeds` | 未设置 | 明确的非连续 seed 列表，例如 `0,7`；与 `--tasks` 互斥，不能搭配非零 `--seed-start` |
| `--hours` / `--horizon-steps` | `24` | 每个任务的模拟经营小时数，正整数 |
| `--k` | `1` | RSI 递归预算，正整数；计划经营执行次数为 `4 × k × tasks` |
| `--mode` | `parallel` | `parallel` / `sequential` |
| `--order-seed` | `0` | 调度种子与 namespace 后缀，不是任务场景 seed |
| `--model` / `--solver-model` | 环境变量或 DeepSeek 默认值 | 经营执行模型 |
| `--improver-model` | 环境变量或所选经营模型 | 提案、合成与经验提炼模型 |
| `--model-config` | `HARNESS2_CLEAN_MODEL_CONFIG` 或 `config/ecom-models.json` | OpenCode 模型配置文件，不将文件内容发给浏览器 |
| `--timeout` | `1800` | **单次经营 rollout 的真实秒数**，不是模拟小时，也不是整批超时 |
| `--output` | 自动创建 | 新实验目录；恢复/查看时指定已有目录 |
| `--port` | `8765` | 看板回环端口；`0` 让系统选择空闲端口 |
| `--no-dashboard` | 关闭 | 只启动训练，以后可以单独 `serve` |
| `--dry-run` | 关闭 | 仅预览配置与任务顺序，零文件写入、零模型调用 |

默认经营模型与改进模型为 `ecom_deepseek/DeepSeek-V4.1-Flash`，支持原有
`HARNESS2_SOLVER_MODEL` / `HARNESS2_IMPROVER_MODEL` 环境变量。
模型 ID 必须已在所选配置文件中定义。API 密钥仍由现有配置读取，不增加命令行密钥参数。

`--timeout` 对应 `HARNESS2_ECOM_TIMEOUT`，不修改现有改进器的超时/重试策略。
实际模型调用数大于经营执行次数，还包括提案、合成、经验提炼及可能的重试。
扩大任务数、小时数或 k 前应重新估算预算。

### 动作配额与定点验证

固定桥接针对已锁定的 public default scenario，遵守原生每模拟步 **30 次 `/act` turn** 的限制。
查询同样计入 turn，批量动作按请求计 turn，不按批内商品数量计。
桥接会通过公开 observation 读取真实 `turn_count`，为显式 `end_of_step` 预留最后一次额度；
到 29 次时拒绝普通动作并提示先结束该步，不自动提交 `end_of_step`、不提高原生额度。
多进程桥接调用使用同一工作区文件锁，避免并发用尽最后一格。桥接不管理绕过它的直接 SDK 调用；
若 SDK 已耗尽全部额度，将明确报告 `turn_quota_exhausted`，等待或重新 observe 不会重置该步。
每次 observe/result 都会提示 `action_budget`，配额拒绝不会伪装成网络限流。

只验证 seed0、seed7，而不误跑中间 seed：

```bash
sh scripts/ecom-train.sh run --seeds 0,7 --hours 24 --k 1 --mode parallel \
  --order-seed 0 --timeout 1800 --port 0 --dry-run
# 真实启动时去掉 --dry-run；仍需确认费用预算。
```

修改固定桥接会改变 experiment identity，必须新建目录并重新运行对应基线，
不能把新协议结果填回旧训练、混用旧缓存，或把定点故障验证称作独立泛化评估。

## 查看、停止、恢复

```bash
# JSON 状态快照，零模型调用
sh scripts/ecom-train.sh status --output runs/<实验目录>

# 单独查看，前台服务；Ctrl+C 只停止此看板，不停止训练
sh scripts/ecom-train.sh serve --output runs/<实验目录> --port 18766

# 或单独后台启动看板；如本目录已存在看板则返回它的 URL
sh scripts/ecom-train.sh serve --output runs/<实验目录> --port 0 --background

# 停止 worker 与该次 sweep 的子进程；看板继续保留
sh scripts/ecom-train.sh stop --output runs/<实验目录>

# 检查状态已中断后恢复。这会继续产生模型费用。
sh scripts/ecom-train.sh resume --output runs/<实验目录> --port 18766
```

- OS 文件锁防止同目录重复启动；不会只凭旧 PID 认定任务存活。
- 恢复复用原有 sweep 的 summary/cache/experience 机制，并重新执行 judge/report/验收。
  这不是模型生成过程的断点续传，中断阶段可能重新调用模型。
- 训练清单、配置文件 hash 或 runner/环境策略源码发生变化时拒绝恢复。任务数、seed、
  模拟小时数、模型等改动必须新建实验，不要手工修改 `train.json` / `tasks.json`。
- `stop` 是异步请求；先用 `status` 确认 `worker_alive: false`，再恢复。
- 如停止后台看板，先核对根目录 `server.json` 的 PID 与进程命令确为本目录的
  `ecom_train serve`，再 `kill <PID>`。不要按名称全局杀进程。
- 端口冲突在任何付费 worker 启动前失败。可换端口或设 `--port 0`。
  若创建目录后预检失败，修好配置后使用 `resume`，不要对同目录再次 `run`。

已有旧实验也可只读查看：

```bash
sh scripts/ecom-train.sh serve \
  --output runs/ecom-rsi-snapshot-fixed-20261021-233057 --port 18765 --background
```

根目录含多个历史 namespace 时增加 `--namespace ecom-oc-agg1`。
旧实验不能直接 `resume` 为本工具的托管训练，避免混淆身份与缓存。

## 页面显示与可信边界

页面约每两秒刷新，按“整批进度、任务队列、所选任务阶段、事件日志”组织：

- 完成/失败任务数、计划/完成经营次数、回退阶段数、本轮启动以来的耗时。
- 基线、INV/DOM 探测、最终执行、经验写回，以及当前改进器调用的最近工具事件。
- 反思状态分别展示：`written` 为经验已写回；`no_entry` 为合法无可迁移经验、无需写回；
  `error` 为反思调用/解析失败；`missing` 为已提炼经验但库产物缺失。旧记录缺少明确状态且
  经验为空时展示 `unknown`，不从空字典猜测 NO_ENTRY，也不伪造经验补齐记录。
- 最后活动时间、长时间无新事件提示，和独立的看板断线状态。
  看板已连接不等于训练正在运行；长时间无日志也不等于失败。
- 评分后显示基线/最终净资产、差值、最终利润与经营存活；评分前标记“未评分”。
  `k > 1` 时基线资产取均值，最终资产取 selected。正差值不证明统计显著。
- 选择任务后可看经营日志；来源也包括整批日志、报告及正在运行的改进器。
  “自动跟随：关”停止自动滚动，仍继续刷新日志与状态。
- 仅展示 48 KiB 原始日志尾部；JSONL 最多格式化尾部的 80 个事件，省略长工具内容。
  不加载完整模型推理。最新状态不以浏览器自动重试模型的方式获得。
- 手机窄屏改为上下布局，表格内部横向滚动；支持键盘焦点与减少动效偏好。

服务**仅绑定 `127.0.0.1`**，无公网监听参数、外部 CDN、遥测、启动/修改 HTTP 接口。
API 只有 `GET /api/status` 和 `GET /api/log`；拒绝跨来源 Host/Origin，拒绝写方法，
日志来源使用白名单且校验路径边界，不提供目录或任意文件下载。日志会尽力遮盖
已知凭据、Bearer、常见密钥字段与带密码 URL，但**不能保证模型日志绝无敏感内容**。
不要开放端口、反向代理公网或将运行日志作为安全脱敏后的可公开数据。

这不是安全沙箱。训练仍使用原有 clean 环境策略，必须另行管理本机权限与网络边界。

## 产物和完成判断

根目录：

```text
train.json            冻结的训练参数与输入 hash
tasks.json            MerchantBench 原生格式任务清单
job-status.json       原子写入的当前阶段、状态、PID、时间与阶段退出码
job-history.jsonl     历次 resume 前的状态快照
worker.lock           worker 运行锁（锁是否被占用比文件存在更重要）
run.log               check/run/judge/report 合并输出，恢复时追加
worker.log            detached worker 的未捕获异常等辅助输出
verification.json     任务、回退、经营产物完整性与评分报告验收
report.txt            最新报告阶段输出
server.json           后台看板端口与 PID（本身不能证明进程在线）
dashboard.lock        看板运行锁
dashboard.log         看板辅助日志
```

正常训练依次执行：`check → run → judge selected → judge --bases → report → verify`。
**report 没有 `--bases` 参数**，工具不会传入该错误参数。
验收检查所有任务成功、无改进阶段回退、全部 `4 × k × tasks` 次经营产物存在且
fingerprint 完整、selected/base 评分齐全、报告存在，以及每个任务反思完成。
反思完成指经验有效且实际写回库，或明确记录合法 `NO_ENTRY`；后者不要求生成经验文件。
调用失败、解析失败、写回缺失和无法判定的旧记录都不能冒充合法弃权。

worker 返回零并记录 `completed` 仅在验收全部通过时发生。
出现回退或验收不完整时记录 `completed_with_warnings`；进程或命令错误记录 `failed`；
停止记录 `interrupted`。启动命令返回零只代表已启动 detached worker，
**不能把它的退出码当作训练成功证明**，应查看 `status` 与 `verification.json`。

## 离线验证

```bash
.venv/bin/python -m unittest discover -s tests -p test_ecom_train.py -v
.venv/bin/python -m unittest discover -s tests -v
```

新增测试不调用模型，覆盖参数、顺序、不可变输入、重复启动锁、完整命令链、
sequential/k>1、失败/回退、产物篡改、历史状态、实时阶段、日志边界/脱敏及 HTTP 只读约束。
`tests/test_ecom_reliability.py` 还覆盖最后动作预留、过期决策、并发桥接、上游真实 30 次限制、
显式 end_of_step 到 terminal，以及 NO_ENTRY/传输失败/格式失败的持久化区分。
浏览器验证可直接使用已完成历史实验，不需要重新付费训练。
