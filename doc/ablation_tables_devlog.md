# 表 3 / 表 4 配置与自动评估工作记录

## 2026-10-20

本轮用户明确授权读取 `ApDepth-G-paper/ieee-dc-template.pdf`，核对两个消融表，补齐配置，
提供一张表一个训练后自动评估脚本，并把现有 `script/eval/0_infer_sky.sh` 补成 inference+eval 入口。
论文仅只读；临时提取和渲染文件放在 `tmp/pdfs`，没有编辑论文。

| 阶段 | 编写/核验状态 |
|---|---|
| PDF 第 9/10 页文本提取、PNG 视觉核对 | 完成 |
| 表 3/4 配置映射与缺项 | 完成，11 个保留、4 个新增 |
| 真 8ch、训练专用 DA2、独立 fade 开关 | 完成实现，运行验证待执行 |
| checkpoint 自带 config/scheduler 组件加载 | 完成实现，运行验证待执行 |
| 统一 sky inference+eval 和两张表的运行器 | 完成实现，运行验证待执行 |
| Linux 命令、恢复/失败处理说明、静态检查 | 完成 |
| 单元测试、真实训练/推理、实测表格 | 未执行/待服务器运行 |

本轮代码与文档编写进度 100%；这不表示训练、数值验证或论文结果已完成。

### 论文与源码差异

- Table III：00、02、01、09、10、08 已有；缺真 8ch 无 VGC、8ch+VGC。
  新增 `11_8ch_no_vgc.yaml`、`12_8ch_vgc.yaml`，而不是把 ID 08 的 12ch 零通道模式改称 8ch。
- Table IV：00、03、04、05、06、07 已有；新增 N1（只取消 SNR floor）、N2（只取消 terminal fade）。
- 旧 runner 只做 NYUv2/KITTI 常规指标，不够填新版 Table III 的 3×3 天空列、
  也缺 Table IV 的 DIODE Outdoor IQR/Ord.，因此保留为 legacy 工具，新增两个表专用运行器。
- PDF 正文训练预算 23000，当前用户本地 `config/train_marigold.yaml` 已修改为 8000，
  且基础模型目录名为 `stable-diffusion-2`。保留原文件、预算、LR 长度和目录名，不回滚用户改动。
  所有消融继承同一主配置；复用 Full 时检查其保存配置和预算匹配。

### 实现要点

- 8ch 真正输入 RGB+noisy-depth；由原 SD2 4ch conv_in 重复两份/2，bias 复制。
  12ch 原默认仍重复三份/3。拒绝把已训练的 12ch 模型裁成 8ch。
- 8ch+VGC 训练保持原来的三个 VGC 项和支撑区定义，只拿掉 U-Net 的显式 prior 通道。
  8ch 推理不加载 DA2 权重或执行 DA2 forward；无 VGC 的 8ch 训练也不加载 DA2。
- N2 使用新开关 terminal_noise_fade_enabled=false，与终端高斯匹配分开。
  精确 T-1 处 Gaussian matching 在 offset 添加后再次保证；默认 Full 在该处原本 fade=0，数值语义保持。
- run.py / infer.py 在 pipeline 构建之前传入该 checkpoint 的完整 U-Net/scheduler 组件，
  `from_pretrained(<checkpoint>/unet)` 读取自己的 config.json，scheduler 读取自己的 scheduler_config.json。
  不复制/改写 BASE_CKPT_DIR；记录哈希并检查配置匹配。
- 每个实验 train → 最终 step checkpoint 完整性检查 → 测试 → 汇总 → 下一个实验。
  train.py 某些异常路径会返回 0，运行器额外要求真正的最终 checkpoint，不能把暂停的 latest 当结束。
- 天空入口复用冻结 split/mask，必要时从既有 DIODE all mask 中派生 Outdoor 索引，无需重新分割。
  记录祖先 manifest/split 哈希；天空与 conventional benchmark 使用不同预测目录和图像坐标配置。
- Table III 三数据集 IQR/Wide/SFOE；Table IV NYUv2/KITTI AbsRel×100、δ1%，及 DIODE Outdoor IQR/SFOE。
  每个指标跨行使用共同有效样本，核对 RGB、mask 和 protocol 一致，输出有效 N。
  mask 导致的边界不足/空内部保留排除记录；缺预测、非有限值、shape 错误、退化前景范围停止成功发布。
- 可 --dry_run、--eval_only、--resume、--collect_only；失败产物保留，不自动覆盖旧 prediction/metrics。

### 实际检查与限制

已执行：

- `pdftotext -layout` 提取本地 PDF；`pdftoppm -f 9 -l 10` 渲染并目视核对 Table III/IV。
- 标准库 AST 解析 16 个新增/修改 Python 文件；通过。
- 静态合并/检查 15 个 YAML 配置：全部继承 max_iter=8000；两个新8ch、N1、N2 取值符合预期。
- Git Bash `bash -n script/eval/0_infer_sky.sh`；通过。
- 对本轮 tracked 变动执行 `git diff --check`；未发现空白错误（Git 提示部分文件下次操作会转换行尾）。

新增 `tests/test_table_ablation.py` 的 5 个无下载 fixture 用例：表格行归属、8ch 初始化与拼接、
缺失配置阻断、checkpoint 本地组件路径/基础文件不变、scheduler 不匹配阻断；
原消融测试更新了 15 个配置的矩阵及用户可控预算约束。
沿用用户之前“先不用测试”的要求，**这些用例均未执行**。
没有训练、下载模型/数据集、运行真实指标、填论文成绩、提交或推送。

运行方法见 `config/ablation/README.md`。后续仍需在服务器用真实 checkpoint 验证推理和数值结果，
并在论文中统一最终采用的训练预算；本轮不修改 PDF 或论文源码。
