# SCOE 开发日志

## 2026-10-19：P0 完成（10%）

| 阶段 | 权重 | 状态 |
|---|---:|---|
| P0 源码审计 | 10% | 完成 |
| P1 真实数据与划分 | 20% | 名单静态审计已完成；真实数据审计待补充（计 5%） |
| P2 掩码与配对 | 20% | 代码、缓存、QA 图实现；运行验收待执行（计 15%） |
| P3 指标与汇总 | 20% | 代码与性质测试用例已实现；未运行（计 15%） |
| P4 CLI 与导出 | 15% | 四阶段入口及导出已实现；端到端验收待执行（计 10%） |
| P5 测试与文档 | 15% | 文档及测试文件已交付；未执行测试（计 5%） |

按原任务含真实数据与验证的完整交付口径，保守总进度约 **60%**；
这不是测试通过率，也不是已完成实验的比例。本轮用户缩小为先交付实现，暂不测试。

初始 commit：`02e2571257fbf2bfcc4655e183793255023a3746`，分支 main。
初始非论文工作区变更：用户已有 `src/__pycache__/__init__.cpython-312.pyc` 修改和未跟踪 `agent.md`，保留。
全部搜索、读取、输出排除 `ApDepth-G-paper`，不读取其中内容。

已运行：限定路径的 `git status --short`、`git diff --stat main`、`git rev-parse HEAD`；
阅读 README、run/infer/eval、dataset、util、配置和已有推理/评估测试。
Python 标准库审计仓库 VKITTI 名单：train=20148（Scene02/06/18/20），val=5364（Scene01）；
scene、精确 scene/variation/camera/frame 交集均为空。训练/验证均含两个 camera、六个 variation。
`F:/train_marigold/config.yaml` 引用相同相对训练名单，但未提供当时名单快照或目标 checkpoint，故不能证明目标模型独立。
扫描 `F:/vkitti` 未找到 `colors.txt`、`classgt_*.png`；有 RGB/depth 和单独权重文件。
已检查 F 盘常见数据目录，未定位 HRWSI，不能推断整机不存在该数据集。
默认 Python 缺 numpy；`D:/Anaconda/envs/DL/python.exe` 已有 numpy/scipy/Pillow/PyYAML，可用 unittest 测试，无需安装。
本阶段未运行数值测试。

### 实现路径映射（一页内）

| 原接口 | 复用方式 / 限制 | 新模块 |
|---|---|---|
| run.py | `depth_npy/<RGB stem>_pred.npy`；多场景同名帧必须分目录运行或显式映射，防碰撞 | predictions.py |
| infer.py | 保留 RGB 相对父目录；VKITTI id 模式为 `pred_rgb_00000.npy` | predictions.py |
| VKITTI dataset | 原配置启用 benchmark crop，SCOE 必须用完整 RGB 坐标；拒绝把裁剪预测拉伸成全图 | datasets.py、schemas.py |
| eval.py | 含 GT 对齐和裁剪，不调用其处理后预测 | metrics.py、aggregate.py |
| model_overrides.py | 本地仅接受 12ch U-Net，必须匹配 scheduler 和 inference_settings/config；不假定已支持 8ch | 预测元数据记录 |
| marigold_pipeline.py | 输出前 clip 到 [0,1]；记录 native postprocessing，不再额外 clip，不宣称未截断 latent | 预测元数据记录 |
| data_split/config | 审计文件哈希、scene 和精确帧交集；目标训练证据另行提供 | audit.py |
| 新入口 | audit → prepare → evaluate → report，CPU 离线、冻结配置/数据/配对、失败计数 | script/sky_eval.py、src/sky_eval/ |

不更改旧训练、推理、eval、README 或用户已有文件；使用独立中文指南。

## 2026-10-19：P1–P5 实现交付，运行验证暂缓

用户补充：所需语义标注、HRWSI、目标预测及训练记录均未准备，要求“先不用测试，先完成这些东西”。
按此要求完成实现与测试文件，**没有运行单元测试、性质测试、CLI 烟雾测试、真实数据 evaluator 或模型推理**。
没有下载数据集、模型或依赖，没有提交、推送。

新增实现：

- `schemas.py`：协议和 manifest 类型、相对路径约束、哈希与拒绝覆盖写入。
- `datasets.py` / `audit.py`：显式 HRWSI manifest、VKITTI 官方 RGB palette 校验、未知标签拒绝、RGB/mask 对齐、
  5% 筛选、全天空排除、精确帧/场景交集、字节重复检查、逐样本失败记录。
- `masks.py`：内部 EDT、逐侧 2% 候选带、法线朝前景、模型无关栅格采样与缓存、半射线类别检查。
  静态审查发现统一向上舍入会让前景侧偏移一像素，改为半像素平局朝边界中点舍入，并检查实际 EDT 取样带。
  这是一项未跑模型前的离散实现修正，未改 5%/2% 参数。
- `metrics.py` / `aggregate.py`：IQR/Wide/SFOE、分母退化、指标独立缺失、逐图等权、固定共同交集；
  对偶数中位数采用先除二再相加，避免有限大数求和溢出。
- `predictions.py`：原 run/infer 命名与显式映射；拒绝命名碰撞、非法 dtype/shape/NaN/Inf；
  明确方向、原生后处理、完整图像坐标；不执行重采样或非线性表示转换。
- `workflow.py` / `script/sky_eval.py`：audit/prepare/evaluate/report、dry-run、limit、退出码、run/model 隔离；
  冻结配置、源码、审计、名单、数据、pair cache 与结果，阻断过期结果或数据漂移。
- `visuals.py`：掩码/内部/候选带/法线 QA，低/高排序错误案例及共享显示范围模型对比。
- `config/sky_eval.yaml`：默认 5%/1%/2%/1%、50 对、linear、禁止场景泄漏；真实路径留空。
- `doc/sky_eval.md`：中文指南、公式、模型元数据、复现命令、数据依赖及当前限制。
- `tests/sky_eval/test_scoe.py`：17 个 unittest 方法，包括四张人工 RGB/mask/float32 预测的两模型端到端流程。
  官方 palette 回归是合成样例，真实官方标注回归仍待数据到位。

实际执行检查：

1. 使用 Python 标准库 `ast.parse` 对 12 个新增 Python 文件做语法解析；通过。
2. AST 检查新增函数的参数和返回类型注解；未发现缺失（self/cls 除外）。
3. 新增 Python 文件尾部空白静态扫描；通过。
4. 限定目录的 `git diff --check`；无输出。它不覆盖 untracked 新文件，所以另做上述空白扫描。
5. 限定目录的 `git status --short`；原 tracked 变更仍仅为用户已有 pyc，本轮其余均为新增文件。
6. 用标准库重新统计仓库 train/val 名单并写 `doc/sky_eval_repository_audit.json`，包含名单哈希和静态检查状态。

未使用测试结果支撑任何数值正确性声明。语法解析通过不等于运行或数值验证通过。

仍待完成：用户允许后运行测试并修复问题；补齐真实语义标注与 HRWSI；目标 checkpoint 训练历史核验；
真实 mask 至少 50 张 QA、配对保留率审查、模型输出表示审查及全量评估。
正式表格目前无数据；默认配置中 models 为空，不编造论文结果。
