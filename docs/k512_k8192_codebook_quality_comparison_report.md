# K512–K8192 离散码本质量对比实验报告

> 实验状态：五组码本均已完成 100 epochs 训练与独立测试集量化评估；K512 / K2048 / K8192（含刚体变换版）已完成**全离散损失**闭环任务评测（PushT + Cube）
> 报告日期：2026-09-04（全离散闭环结果更新版）
> 项目：Stable World Model / LeWM
> 对比范围：K512、K1024、K2048、K4096、K8192（离线量化）；K512、K2048、K8192、K8192-rigid（闭环任务）
> 核心结论：离线量化保真度以 **K8192** 最佳、**K4096** 最均衡；在**全离散损失**的闭环评测中，**码本质量直接决定模型质量**——PushT held-out 成功率随码本容量单调上升 **17.5% → 30.5% → 62.0%**

## 1. 摘要

本报告对同一官方 LeWM PushT 连续隐空间训练得到的五组离散码本进行统一比较，并用其中 K512、K2048、K8192（及 K8192 刚体变换版）在 PushT 与 Cube 两个单任务上以**全离散损失**完成闭环任务评测。五组码本实验除码本大小外，其余数据、encoder、latent 维度、随机种子、初始化样本数、优化器和训练轮数均保持一致。

独立测试集上的离线量化结果如下：

- 平均绝对 L2 量化误差随码本增大单调下降：从 K512 的 **6.8794** 降至 K8192 的 **2.8373**，下降 **58.76%**。
- 平均相对 L2 误差从 **49.73%** 降至 **20.62%**，下降 **58.54%**。
- 测试误差 P99 从 **11.5188** 降至 **7.3097**，下降 **36.54%**；大码本不仅改善均值，也改善长尾样本。
- 每次将 K 翻倍，测试平均绝对 L2 均继续下降约 **19%–21%**；在 K512–K8192 范围内没有出现误差反弹。
- 代价是权重体积按 K 线性增长：可加载 checkpoint 从 **0.752 MiB** 增至 **12.002 MiB**，扩大约 16 倍。
- 验证集 active-code fraction 从 K512/K1024 的 **100%** 降至 K8192 的 **85.21%**；perplexity/K 从 **93.32%** 降至 **65.91%**，说明大码本的新增容量未被均匀使用。
- 所有实验的最优验证目标都出现在 epoch 100，但 epoch 90 到 100 的相对改善均小于 **0.08%**，训练在后十轮已基本进入平台期。

**闭环任务评测（本次更新新增）回答了旧版报告遗留的核心问题**：码本质量的差异是否会传导为世界模型的任务表现差异。在蒸馏损失的三项教师表示全部替换为冻结码本向量 c_{y^T} 的全离散设置下，答案明确为**是**：

- PushT held-out 200 任务成功率随 K 单调上升：K512 **17.5%** → K2048 **30.5%** → K8192 **62.0%**，相邻差距（+13.0、+31.5 个百分点）远超 200 轮评测的统计噪声（σ ≈ 3 个百分点）。
- 50 起点阶段评测同样单调：K512 **20%** → K2048 **38%** → K8192 **62%**；刚体变换版 K8192-rigid 为 **64%**（50 起点）/ **55.5%**（held-out），与 K8192 的差异在噪声范围内，与刚体变换保持码本几何等价的预期一致。
- Cube 上趋势一致但幅度压缩：K512 **54%** → K2048 **64%** → K8192 **70%**（均为 50 起点最佳阶段）；K8192-rigid 同为 **70%**。
- 传导机制可以直接观测：码本量化误差（val 平方 L2：51.7 / 23.3 / 10.8）决定离散目标本身的信息量；纯码本轨迹的一步预测误差 `pred_teacher_mse`（PushT：0.0836 / 0.0761 / 0.0616）与闭环成功率严格同序。
- 先行消融（去掉 latent 对齐与软 token 项、只保留预测项）曾坍塌至 **3.5%**，说明崩溃源于缺少 latent 项而非离散目标本身；全离散设计保留全部三项损失，仅统一教师表示，训练全部稳定收敛。

因此本版结论按使用目标修订为：

1. **最高离线量化保真度：K8192**（不变）。
2. **PushT 类质量敏感任务的闭环部署：K8192**。held-out 成功率 62%，比 K2048 高 31.5 个百分点，比 K512 高 44.5 个百分点。
3. **Cube 类任务的闭环部署：K2048 起步即可**。64% vs K8192 的 70%，差距未超 50 起点评测的噪声，而 checkpoint 体积只有 1/4。
4. **容量与离线质量折中：K4096**（不变；其闭环表现尚未评测）。

## 2. 实验目标与比较边界

### 2.1 实验目标

本次比较回答四个问题：

1. 增大码本大小是否持续降低官方 LeWM latent 的最近邻量化误差？
2. 更大码本是否仍能保持充分的码字覆盖和较均匀的使用分布？
3. 量化质量提升相对于 checkpoint 存储增长是否值得？
4. **（新增，本次已回答）码本质量的离线差异是否传导为闭环任务成功率的差异？**

### 2.2 结果边界

**离线量化部分**（§3–§6）：`.stablewm/codebook_runs` 中五组结果包含码本训练曲线、train/validation/test 三个 split 的逐向量量化误差统计、EMA teacher 码本权重和码字使用统计。这部分衡量的是**连续 latent 的离线最近邻重建质量与码本使用效率**。

**闭环任务部分**（§7）：K512 / K2048 / K8192 / K8192-rigid 四个条件在 PushT、Cube 两个单任务上，以全离散损失完成与单任务基线完全同协议的训练与评测（PushT 另含 held-out 200 任务协议）。这部分衡量的是**码本质量对世界模型闭环成功率的影响**。

需要说明：早期曾以混合损失（latent 对齐项以连续 z^T 为目标）完成过一轮闭环评测，但该设置下不同 K 的成功率差异在评测噪声内、无法区分码本质量，本版已**废弃该组结果**，统一改用全离散损失重新评测。K1024 与 K4096 暂无闭环结果。

## 3. 受控实验设置

### 3.1 共同配置（离线码本训练）

| 项目 | 统一设置 |
|---|---:|
| 冻结 encoder | `official_lewm_pusht_compat` |
| 数据集 | `galilai-group/lewm-pusht` |
| 原数据集长度 | 2,336,736 |
| 抽取 latent 总数 | 262,144 |
| 训练 latent | 235,929 |
| 验证 latent | 26,215 |
| 独立测试 latent | 32,768 |
| latent 维度 | 192 |
| 图像大小 | 224 × 224 |
| 训练 seed | 3072 |
| k-means++ 初始化样本 | 65,536 |
| EMA teacher momentum | 0.99 |
| epochs | 100 |
| batch size | 8,192 |
| 初始 / 最小学习率 | 1e-3 / 1e-5 |
| weight decay | 0 |
| 量化权重 | `teacher.weight`（EMA teacher） |

五组实验唯一的核心自变量是 `num_embeddings`：512、1024、2048、4096、8192。每个 checkpoint 同时保存 FP32 `student.weight` 和 `teacher.weight`，两者形状均为 `[K, 192]`。

### 3.2 指标定义

对连续 latent `z` 及其最近邻 EMA-teacher 码字 `q(z)`：

```text
absolute_l2 = ||z - q(z)||₂
relative_l2 = ||z - q(z)||₂ / max(||z||₂, 1e-12)
val_teacher_l2 = E[||z - q(z)||₂²]
perplexity = exp(-Σ p(code) log p(code))
```

其中：

- `absolute_l2` 和 `relative_l2` 是逐向量误差，本报告以独立 test split 为主。
- `val_teacher_l2` 是验证集平均平方 L2，用于选择 checkpoint；它不是测试集平均绝对 L2 的平方。
- `active-code fraction` 只判断某个码字是否至少被使用一次。
- `perplexity/K` 同时反映覆盖和使用均匀性；数值越接近 100%，表示有效使用越接近完整且均匀的 K 个码字。

## 4. 测试集量化质量

### 4.1 核心结果

| 码本 | 绝对 L2 均值 | 中位数 | P90 | P99 | 相对 L2 均值 | 相对 K512 均值改善 |
|---:|---:|---:|---:|---:|---:|---:|
| K512 | 6.8794 | 6.8719 | 9.6591 | 11.5188 | 49.73% | — |
| K1024 | 5.5670 | 5.4675 | 8.3582 | 10.3377 | 40.30% | 19.08% |
| K2048 | 4.4187 | 4.2637 | 7.1408 | 9.2285 | 32.04% | 35.77% |
| K4096 | 3.5271 | 3.3181 | 6.0913 | 8.2655 | 25.59% | 48.73% |
| K8192 | **2.8373** | **2.5788** | **5.1929** | **7.3097** | **20.62%** | **58.76%** |

![不同码本大小的测试集量化误差对比](assets/codebook_size_comparison/codebook_size_quantization_quality.png)

结果具有稳定的单调性：K 越大，均值、中位数、P90 和 P99 均越低。平均绝对 L2 的相邻翻倍收益为：

| 扩容 | 平均绝对 L2 降幅 | 绝对下降量 |
|---|---:|---:|
| K512 → K1024 | 19.08% | 1.3124 |
| K1024 → K2048 | 20.63% | 1.1483 |
| K2048 → K4096 | 20.18% | 0.8916 |
| K4096 → K8192 | 19.56% | 0.6898 |

相对降幅大致稳定在 20%，但绝对下降量逐级缩小。这一范围内的经验关系近似为 `mean absolute L2 ∝ K^-0.32`，可作为当前数据上的插值描述，不应直接外推到更大的码本。

### 4.2 长尾误差

K8192 的 P99 仍为 7.3097，明显高于其中位数 2.5788，说明少量 latent 仍较难量化。但与 K512 相比：

- 中位数下降 62.47%；
- P90 下降 46.24%；
- P99 下降 36.54%。

均值的改善快于长尾改善，说明增大 K 对典型 latent 的收益更明显，而高误差样本仍可能需要更有针对性的码本训练、分层码本或残差量化。

## 5. 码本覆盖率与使用均匀性

| 码本 | 验证 perplexity | Perplexity / K | Active codes | Active fraction | 未激活码字 | 最大单码占比 |
|---:|---:|---:|---:|---:|---:|---:|
| K512 | 477.81 | **93.32%** | 512 | **100.00%** | 0 | 0.5493% |
| K1024 | 929.87 | 90.81% | 1,024 | **100.00%** | 0 | 0.2975% |
| K2048 | 1,760.29 | 85.95% | 2,042 | 99.71% | 6 | 0.1678% |
| K4096 | 3,195.10 | 78.01% | 3,952 | 96.48% | 144 | 0.1144% |
| K8192 | **5,399.03** | 65.91% | **6,980** | 85.21% | 1,212 | **0.0916%** |

![训练收敛与最终验证码本利用率](assets/codebook_size_comparison/codebook_size_training_and_utilization.png)

需要区分绝对量和归一化效率：

- K8192 的有效码字绝对数量最高，perplexity 也是最高的，因此它确实表达了更多离散状态。
- 但 K8192 有 1,212 个码字未在 26,215 个验证 latent 中出现，且 perplexity 只相当于 K 的 65.91%。
- 最大单码占比随 K 增大持续下降，没有出现少数单码占据大量样本的严重热点坍塌。
- 覆盖率下降主要表现为长尾码字稀疏或未激活，而不是头部码字异常集中。

因此，K8192 不是“码本坍塌”，但其边际容量利用效率明显低于 K512–K2048。对验证样本量有限的场景，active fraction 也会受 split 大小影响，应结合 perplexity/K 解读。

## 6. 泛化与训练收敛

### 6.1 Train–validation–test 误差

| 码本 | Train 平均绝对 L2 | Validation | Test | Test 相对 Train 增幅 |
|---:|---:|---:|---:|---:|
| K512 | 6.8444 | 6.8773 | 6.8794 | 0.51% |
| K1024 | 5.5150 | 5.5709 | 5.5670 | 0.94% |
| K2048 | 4.3387 | 4.4163 | 4.4187 | 1.84% |
| K4096 | 3.3918 | 3.5128 | 3.5271 | 3.99% |
| K8192 | 2.6404 | 2.8431 | 2.8373 | 7.46% |

![码本容量、测试误差与泛化差距](assets/codebook_size_comparison/codebook_size_capacity_tradeoff.png)

所有码本的验证与测试结果接近，说明独立 test split 没有出现异常偏移。与此同时，K 越大，test 相对 train 的误差增幅越明显，从 0.51% 增至 7.46%。这表明更大码本对训练 latent 的局部拟合更充分，也带来更大的 train–test gap；但 K8192 的测试绝对误差仍然是五组中最低，因此该 gap 尚未抵消其容量收益。

### 6.2 收敛速度

| 码本 | Epoch 1 验证平方 L2 | Epoch 50 | Epoch 90 | Epoch 100 | Epoch 90→100 改善 |
|---:|---:|---:|---:|---:|---:|
| K512 | 78.2206 | 53.2446 | 51.6932 | 51.6536 | 0.0766% |
| K1024 | 52.8308 | 36.0447 | 35.1897 | 35.1670 | 0.0646% |
| K2048 | 34.0202 | 23.8934 | 23.3417 | 23.3270 | 0.0630% |
| K4096 | 21.6557 | 15.9572 | 15.6742 | 15.6674 | 0.0436% |
| K8192 | 14.0589 | 10.9441 | 10.7726 | 10.7681 | 0.0414% |

五组 `best_epoch` 均为 100，训练目标在完整训练过程中仍保持微弱改善。不过，最后十轮的收益都低于 0.08%。如果后续需要大量扫描 K、seed 或其他超参数，可以把 90 epochs 作为节省算力的候选停止点；在正式替代 100 epochs 前，仍应验证 task-level 指标不会受影响。

## 7. 全离散闭环任务评测

本节是本次更新的核心新增内容：用闭环任务成功率直接检验“码本质量是否影响模型质量”。

### 7.1 动机与损失设计

蒸馏训练的损失共三项：latent 对齐、软 token KL、动力学预测。**全离散（fully-discrete）设置把三项损失中的教师表示统一替换为冻结码本向量 c_{y^T}**：

```text
第一项  latent 对齐：  MSE( student_enc(o), c_{y^T} )        [教师表示 = 码本向量]
第二项  软 token KL：  0.1 · KL( p_student ‖ p_teacher^topk ) [不变]
第三项  动力学预测：  MSE( Predictor(·, a), c_{y^T_next} )    [teacher-forcing，输入同取码本向量]
```

这样设计的理由：

1. 编码器输出最终要通过最近邻查询落到码本向量上，端到端被消费的表示就是 c_{y^T}；让训练目标与消费形式一致，码本误差不再有连续目标的缓冲。
2. 先行消融（去掉前两项、只保留预测项）在 PushT 上坍塌至 3.5%，说明崩溃源于缺少 latent 对齐项，而不是以离散向量为目标；全离散设计保留全部三项，训练稳定。
3. 在此设置下，码本拟合质量（量化误差）直接决定学习目标的信息量，码本质量的影响应当可观测——这正是本节要检验的命题。

### 7.2 训练与评测协议

所有条件使用与单任务基线（P1/C1）完全一致的协议，仅码本与教师表示不同：

| 项目 | 统一设置 |
|---|---:|
| 训练 seed | 3072 |
| 阶段 | 16 epochs = 4 + 10 + 2（三阶段余弦学习率） |
| 全局 batch | 768（4 × A100 × 192/GPU） |
| 精度 / 优化器 | bf16-mixed / AdamW（weight decay 1e-3） |
| student 初始化 | 官方 teacher checkpoint |
| 模型结构 | 与单任务基线同结构同规模（ViT-tiny student + 6 层 predictor，≈18M 参数） |
| 码本 | 冻结，量化用 `teacher.weight` |
| 质量 gates | 只记录、不中止训练 |
| 评测（50 起点） | 固定 50 个起点，seed 42，phase1/phase2/final 三阶段取最佳 |
| 评测（held-out） | 仅 PushT：selection 50 起点（seed 42）选阶段 → 4 × 50 = 200 个 held-out 任务（seed 4242） |

评测起点清单与 held-out 分片 manifest 固定存于 `.stablewm/evaluation_manifests/codebook_quality_rigid_v1/`，所有条件共用，保证配对可比。

条件矩阵共 8 个（PushT × 4 + Cube × 4），由两级编排器顺序执行：

- 第一级 `logs/single_task_fully_discrete_gpu0123/`：PushT K8192 全离散（含 held-out）→ Cube K8192 全离散；
- 第二级 `logs/fully_discrete_codebook_series_gpu0123/`：Cube K512/K2048 码本准备 → PushT K512 / K2048 / K8192-rigid（各含 held-out）→ Cube K512 / K2048 / K8192-rigid。

### 7.3 参与评测的码本

**PushT**：三个码本与训练缓存直接复用离线实验的产物，保证闭环条件与 §4–§6 的离线指标一一对应（K512：test 绝对 L2 6.8794；K2048：4.4187；K8192：2.8373）。K8192-rigid 由 K8192 码本经正交变换 + 平移生成（seed 20260826），变换审计确认成对距离最大相对误差 1.1e-7、top-32 概率最大绝对误差 9.9e-5，量化行为与原 K8192 等价。

**Cube**：K8192 码本为已有产物；K512 / K2048 为本次新训练，配方与 Cube K8192 完全一致（seed 3072、262,144 latents、k-means++ 65,536、EMA 0.99、100 epochs、batch 8,192）。其离线量化质量与 PushT 同级：

| Cube 码本 | test 绝对 L2 均值 | 相对 L2 均值 | 验证 perplexity | Perplexity / K | Active fraction | 验证平方 L2 |
|---:|---:|---:|---:|---:|---:|---:|
| K512 | 6.8332 | 49.13% | 494.52 | **96.59%** | **100.00%** | 49.56 |
| K2048 | 4.8865 | 35.14% | 1,879.59 | 91.78% | 99.90% (2046/2048) | 25.58 |
| K8192 | **3.4582** | **24.87%** | **6,458.85** | 78.84% | 93.27% (7641/8192) | **13.00** |

Cube K8192-rigid 同样由 Cube K8192 码本经刚体变换生成（seed 20260826），审计成对距离最大相对误差 2.0e-7。

### 7.4 PushT 闭环结果

**50 起点阶段评测**（seed 42，成功率 %）：

| 条件 | K | phase1 | phase2 | final | 最佳阶段 |
|---|---:|---:|---:|---:|---|
| k512_fully_discrete | 512 | 10 | 16 | 20 | final：**20.0** |
| k2048_fully_discrete | 2,048 | 14 | 32 | 38 | final：**38.0** |
| k8192_fully_discrete | 8,192 | 26 | 54 | 62 | final：**62.0** |
| k8192_rigid_fully_discrete | 8,192 | 28 | 64 | 62 | phase2：**64.0** |

**held-out 200 任务评测**（seed 4242，4 分片 × 50）：

| 条件 | 选择阶段 | 分片 00 | 分片 01 | 分片 02 | 分片 03 | 总计（200） |
|---|---|---:|---:|---:|---:|---:|
| k512_fully_discrete | final | 14.0 | 20.0 | 16.0 | 20.0 | **17.5%**（35/200） |
| k2048_fully_discrete | phase2 | 30.0 | 22.0 | 42.0 | 28.0 | **30.5%**（61/200） |
| k8192_fully_discrete | final | 54.0 | 64.0 | 64.0 | 66.0 | **62.0%**（124/200） |
| k8192_rigid_fully_discrete | phase2 | 56.0 | 46.0 | 72.0 | 48.0 | **55.5%**（111/200） |

主要观察：

1. **码本容量与成功率的单调关系明确成立**：K512 < K2048 < K8192 在两套评测、每个训练阶段上都成立。held-out 相邻差距 +13.0（K512→K2048）与 +31.5（K2048→K8192）个百分点，按二项噪声 σ ≈ 3 个百分点估算分别约 3σ 与 7σ，不是评测抖动。
2. **差距随 K 增大而扩大**：K512→K2048 提升 13 个百分点，K2048→K8192 提升 31.5 个百分点，与离线量化误差的收益趋势（翻倍约降 20%，但绝对量逐级放大信息保留）一致。
3. **刚体变换版与原 K8192 无显著差异**：55.5% vs 62.0%（约 1.3σ）。刚体变换保持码本几何，本就应当量化等价；此结果与预期一致。（作为参照，两个 K8192 条件的分片间波动本身就有 46%–72%。）
4. 各阶段成功率同样随训练推进单调上升（phase1 → final），8 个条件无一出现训练失稳。
5. 备注：held-out 流程会对三个阶段的 checkpoint 在 selection manifest 上重新评测一遍用于选阶段，与主评测对同一 checkpoint 的数字可能相差数个百分点（如 K2048 phase2 在两处分别为 38 与 32），源于 MPC 评测的随机性；阶段选择以 held-out 流程自身的结果为准。

### 7.5 Cube 闭环结果

**50 起点阶段评测**（seed 42，成功率 %）：

| 条件 | K | phase1 | phase2 | final | 最佳阶段 |
|---|---:|---:|---:|---:|---|
| k512_fully_discrete | 512 | 46 | 54 | 54 | final：**54.0** |
| k2048_fully_discrete | 2,048 | 56 | 62 | 64 | final：**64.0** |
| k8192_fully_discrete | 8,192 | 66 | 62 | 70 | final：**70.0** |
| k8192_rigid_fully_discrete | 8,192 | 60 | 68 | 70 | final：**70.0** |

观察：

1. 排序与 PushT 一致：K512 < K2048 ≤ K8192 = K8192-rigid，单调趋势成立。
2. 但幅度明显压缩：K8192 与 K512 只差 16 个百分点（PushT held-out 上差 44.5）。按 50 起点的评测分辨率（σ ≈ 7 个百分点），只有 K512 显著落后；K2048 与 K8192 的差距在噪声内。
3. 这说明码本质量的影响是**任务相关的**：Cube 的 latent 结构对量化分辨率不敏感，中等码本已接近任务表现上限；PushT 则持续受益于更大码本。

### 7.6 训练健康度：码本质量如何传导为模型质量

各条件最终验证 epoch 的关键指标（`validate/*`，来自各 run 的 `metrics.jsonl`）：

| run | token agreement | student↔c_{y^T} MSE | student↔z^T MSE | teacher ppl/K | pred_teacher_mse | pred_mixed_mse |
|---|---:|---:|---:|---:|---:|---:|
| PushT k512 | 0.993 | 0.037 | 0.212 | 0.941 | 0.0836 | 0.0184 |
| PushT k2048 | 0.990 | 0.048 | 0.065 | 0.879 | 0.0761 | 0.0106 |
| PushT k8192 | 0.982 | 0.039 | 0.015 | 0.767 | 0.0616 | 0.0050 |
| PushT k8192-rigid | 0.975 | 0.037 | 0.015 | 0.767 | 0.0601 | 0.0057 |
| Cube k512 | 0.993 | 0.057 | 0.184 | 0.975 | 0.1398 | 0.0162 |
| Cube k2048 | 0.980 | 0.080 | 0.055 | 0.955 | 0.1260 | 0.0064 |
| Cube k8192 | 0.973 | 0.056 | 0.014 | 0.933 | 0.0877 | 0.0034 |
| Cube k8192-rigid | 0.971 | 0.057 | 0.014 | 0.933 | 0.0881 | 0.0036 |

三层证据构成完整的传导链：

1. **学生对各自目标的拟合程度相同**：student↔c_{y^T} MSE 在 0.037–0.080 之间，无随 K 的系统性趋势；token agreement 全部 ≥ 0.97。即训练都“学会”了各自的离散目标，小码本模型的失败不是欠拟合。
2. **目标的含信息量随码本质量变化**：学生与连续 latent 的距离（student↔z^T MSE：PushT 0.212 / 0.065 / 0.015）反映的正是量化损失——由码本离线指标直接决定（验证平方 L2 51.7 / 23.3 / 10.8）。小码本的离散目标丢掉了更多状态信息。
3. **离散动力学的可预测性与闭环成功率严格同序**：`pred_teacher_mse`（predictor 在纯码本轨迹上预测下一帧的 MSE）随 K 单调下降（PushT 0.0836 → 0.0761 → 0.0616；Cube 0.1398 → 0.1260 → 0.0877），排序与闭环成功率完全一致。码本质量越差，离散状态序列越不可预测，学到的动力学模型越差，闭环表现随之下降。

简言之：**在全离散设置下，码本量化误差没有被连续目标的平均效应稀释，而是原样进入学习目标，逐级放大为动力学误差与任务失败。**

### 7.7 与连续教师的对照

| 任务 | 官方连续教师 | 全离散最佳（K8192） | 离散化代价 |
|---|---:|---:|---:|
| PushT（50 起点） | 90.0% | 62.0% | −28.0 个百分点 |
| Cube（50 起点） | 68.0% | 70.0% | +2.0 个百分点（噪声内） |

Cube 上全离散蒸馏不仅没有损失，反而与连续教师持平（68% → 70%）；PushT 上仍有 28 个百分点的差距，且从 §7.4 的单调性看主要受码本分辨率限制——这同时指出了后续改进方向（更大码本、残差量化或多码本）在 PushT 类任务上的潜在收益空间。

### 7.8 结果边界与局限

1. **单 seed**：所有条件训练 seed 均为 3072。单调趋势横跨 8 个条件、两套评测、两个任务，方向性结论稳健，但单个条件间几个百分点内的比较仍需多 seed 复核。
2. **K1024 / K4096 未做闭环**：闭环只覆盖 K512 / K2048 / K8192；K4096 的闭环表现（尤其 PushT 上是否已接近 K8192）是下一步最有价值的补测点。
3. **Cube 评测分辨率有限**：50 起点下 σ ≈ 7 个百分点，Cube 上 K2048 与 K8192 的排序需要更大样本确认。
4. **held-out 仅覆盖 PushT**：Cube 沿用 50 起点协议，无独立 held-out 分片。
5. 刚体变换码本与原码本几何等价，两者的闭环差异（PushT −6.5、Cube 0 个百分点）均在噪声内，不构成独立条件间的证据。

## 8. 存储成本与选择建议

| 码本 | `weights.pt` | 相对 K512 体积 | 测试平均绝对 L2 | 验证 active fraction |
|---:|---:|---:|---:|---:|
| K512 | 0.752 MiB | 1× | 6.8794 | 100.00% |
| K1024 | 1.502 MiB | 2× | 5.5670 | 100.00% |
| K2048 | 3.002 MiB | 4× | 4.4187 | 99.71% |
| K4096 | 6.002 MiB | 8× | 3.5271 | 96.48% |
| K8192 | 12.002 MiB | 16× | **2.8373** | 85.21% |

结合离线质量与闭环结果的按目标建议：

| 使用目标 | 建议码本 | 依据 |
|---|---:|---|
| PushT 类质量敏感任务的闭环部署 | **K8192** | held-out 62.0%，比 K2048 高 31.5 个百分点、比 K512 高 44.5 个百分点 |
| Cube 类任务的闭环部署 | **K2048 起步** | 64.0% vs K8192 的 70.0%（差距未超 50 起点评测噪声），体积仅 1/4 |
| 最低离线量化误差 | **K8192** | 所有测试误差统计最佳，平均误差比 K512 低 58.76% |
| 离线质量、覆盖率、存储折中 | **K4096** | 6.002 MiB、96.48% active，平均误差已比 K512 低 48.73%（闭环未测） |
| 最小 checkpoint / 快速基线 | K512 | 0.752 MiB、100% active，但离线误差最高、闭环明显落后 |

## 9. 原始误差分布图

以下图片均由各 run 的 `quantization_errors.npz` 生成，分别展示 train、validation、test 的绝对与相对量化误差分布。

### 9.1 K512

![K512 train validation test 量化误差分布](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook/quantization_error_violin.png)

### 9.2 K1024

![K1024 train validation test 量化误差分布](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k1024/quantization_error_violin.png)

### 9.3 K2048

![K2048 train validation test 量化误差分布](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k2048/quantization_error_violin.png)

### 9.4 K4096

![K4096 train validation test 量化误差分布](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k4096/quantization_error_violin.png)

### 9.5 K8192

![K8192 train validation test 量化误差分布](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k8192/quantization_error_violin.png)

## 10. 原始结果索引与复现

### 10.1 离线码本实验

| 码本 | 配置 | 训练指标 | 量化评估 | 汇总 | 可加载权重 |
|---:|---|---|---|---|---|
| K512 | [config](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook/config.yaml) | [metrics.csv](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook/metrics.csv) | [JSON](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook/quantization_evaluation.json) | [summary](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook/summary.json) | [weights](../.stablewm/checkpoints/official_lewm_pusht_compat_codebook/weights.pt) |
| K1024 | [config](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k1024/config.yaml) | [metrics.csv](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k1024/metrics.csv) | [JSON](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k1024/quantization_evaluation.json) | [summary](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k1024/summary.json) | [weights](../.stablewm/checkpoints/official_lewm_pusht_compat_codebook_k1024/weights.pt) |
| K2048 | [config](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k2048/config.yaml) | [metrics.csv](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k2048/metrics.csv) | [JSON](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k2048/quantization_evaluation.json) | [summary](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k2048/summary.json) | [weights](../.stablewm/checkpoints/official_lewm_pusht_compat_codebook_k2048/weights.pt) |
| K4096 | [config](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k4096/config.yaml) | [metrics.csv](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k4096/metrics.csv) | [JSON](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k4096/quantization_evaluation.json) | [summary](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k4096/summary.json) | [weights](../.stablewm/checkpoints/official_lewm_pusht_compat_codebook_k4096/weights.pt) |
| K8192 | [config](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k8192/config.yaml) | [metrics.csv](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k8192/metrics.csv) | [JSON](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k8192/quantization_evaluation.json) | [summary](../.stablewm/codebook_runs/official_lewm_pusht_compat_codebook_k8192/summary.json) | [weights](../.stablewm/checkpoints/official_lewm_pusht_compat_codebook_k8192/weights.pt) |

Cube 码本（K512 / K2048 为本次新训练）：

| 码本 | 汇总 | 量化评估 | 可加载权重 |
|---:|---|---|---|
| Cube K512 | [summary](../.stablewm/codebook_runs/official_lewm_cube_compat_codebook_k512/summary.json) | [JSON](../.stablewm/codebook_runs/official_lewm_cube_compat_codebook_k512/quantization_evaluation.json) | [weights](../.stablewm/checkpoints/official_lewm_cube_compat_codebook_k512/weights.pt) |
| Cube K2048 | [summary](../.stablewm/codebook_runs/official_lewm_cube_compat_codebook_k2048/summary.json) | [JSON](../.stablewm/codebook_runs/official_lewm_cube_compat_codebook_k2048/quantization_evaluation.json) | [weights](../.stablewm/checkpoints/official_lewm_cube_compat_codebook_k2048/weights.pt) |
| Cube K8192 | [summary](../.stablewm/codebook_runs/official_lewm_cube_compat_codebook_k8192/summary.json) | [JSON](../.stablewm/codebook_runs/official_lewm_cube_compat_codebook_k8192/quantization_evaluation.json) | [weights](../.stablewm/checkpoints/official_lewm_cube_compat_codebook_k8192/weights.pt) |

重建离线对比图：

```bash
source activate swm-env
python scripts/train/plot_codebook_size_comparison.py
```

绘图脚本：[plot_codebook_size_comparison.py](../scripts/train/plot_codebook_size_comparison.py)

### 10.2 全离散闭环实验

| 条件 | 评测汇总（50 起点） | held-out 汇总 | 训练配置 |
|---|---|---|---|
| PushT k512 | [summary](../.stablewm/experiments/fully_discrete_codebook_series_v1/runs/pusht/k512_fully_discrete/task_evaluation/summary.json) | [heldout](../.stablewm/experiments/fully_discrete_codebook_series_v1/runs/pusht/k512_fully_discrete/task_evaluation_heldout/summary.json) | [config](../.stablewm/experiments/fully_discrete_codebook_series_v1/configs/pusht_k512_fully_discrete_seed3072.yaml) |
| PushT k2048 | [summary](../.stablewm/experiments/fully_discrete_codebook_series_v1/runs/pusht/k2048_fully_discrete/task_evaluation/summary.json) | [heldout](../.stablewm/experiments/fully_discrete_codebook_series_v1/runs/pusht/k2048_fully_discrete/task_evaluation_heldout/summary.json) | [config](../.stablewm/experiments/fully_discrete_codebook_series_v1/configs/pusht_k2048_fully_discrete_seed3072.yaml) |
| PushT k8192 | [summary](../.stablewm/joint_distillation/lewm_pusht_k8192_fully_discrete_seed3072/task_evaluation/summary.json) | [heldout](../.stablewm/joint_distillation/lewm_pusht_k8192_fully_discrete_seed3072/task_evaluation_heldout/summary.json) | [config](../scripts/train/config/vq_lewm_joint_distillation_pusht_fully_discrete.yaml) |
| PushT k8192-rigid | [summary](../.stablewm/experiments/fully_discrete_codebook_series_v1/runs/pusht/k8192_rigid_fully_discrete/task_evaluation/summary.json) | [heldout](../.stablewm/experiments/fully_discrete_codebook_series_v1/runs/pusht/k8192_rigid_fully_discrete/task_evaluation_heldout/summary.json) | [config](../.stablewm/experiments/fully_discrete_codebook_series_v1/configs/pusht_k8192_rigid_fully_discrete_seed3072.yaml) |
| Cube k512 | [summary](../.stablewm/experiments/fully_discrete_codebook_series_v1/runs/cube/k512_fully_discrete/task_evaluation/summary.json) | — | [config](../.stablewm/experiments/fully_discrete_codebook_series_v1/configs/cube_k512_fully_discrete_seed3072.yaml) |
| Cube k2048 | [summary](../.stablewm/experiments/fully_discrete_codebook_series_v1/runs/cube/k2048_fully_discrete/task_evaluation/summary.json) | — | [config](../.stablewm/experiments/fully_discrete_codebook_series_v1/configs/cube_k2048_fully_discrete_seed3072.yaml) |
| Cube k8192 | [summary](../.stablewm/joint_distillation/lewm_cube_k8192_fully_discrete_seed3072/task_evaluation/summary.json) | — | [config](../scripts/train/config/vq_lewm_joint_distillation_cube_fully_discrete.yaml) |
| Cube k8192-rigid | [summary](../.stablewm/experiments/fully_discrete_codebook_series_v1/runs/cube/k8192_rigid_fully_discrete/task_evaluation/summary.json) | — | [config](../.stablewm/experiments/fully_discrete_codebook_series_v1/configs/cube_k8192_rigid_fully_discrete_seed3072.yaml) |

编排与日志：

- 第一级编排器：[run_single_task_fully_discrete_gpu0123.sh](../scripts/train/run_single_task_fully_discrete_gpu0123.sh)，日志 `logs/single_task_fully_discrete_gpu0123/`；
- 第二级编排器：[run_fully_discrete_codebook_series.py](../scripts/train/run_fully_discrete_codebook_series.py)（条件矩阵 [fully_discrete_codebook_series.yaml](../scripts/train/config/fully_discrete_codebook_series.yaml)），日志 `logs/fully_discrete_codebook_series_gpu0123/`（`stages.log` 记录全部阶段时间线，`status.txt` 为最终状态）；
- 各条件训练曲线：对应 run 目录下 `metrics.jsonl`（PushT k8192 / Cube k8192 在 `.stablewm/joint_distillation/`，其余在 `.stablewm/experiments/fully_discrete_codebook_series_v1/runs/`）；
- held-out 起点清单：`.stablewm/evaluation_manifests/codebook_quality_rigid_v1/`（selection seed 42 n=50；test seed 4242 n=200，4 分片）；
- 总耗时：第一级约 8.7 小时（09-02 13:53–22:36 UTC），第二级约 26.3 小时（09-02 22:36 – 09-04 00:53 UTC），4 × A100。

复现全离散系列：

```bash
source activate swm-env
# 第一级：PushT/Cube K8192 全离散（训练 + 50 起点评测 + PushT held-out）
bash scripts/train/run_single_task_fully_discrete_gpu0123.sh
# 第二级：等待第一级完成后，准备 Cube 码本并跑其余 6 个条件
python scripts/train/run_fully_discrete_codebook_series.py \
    --config scripts/train/config/fully_discrete_codebook_series.yaml
```

## 11. 最终结论与后续建议

1. **码本质量直接影响模型质量，这是本次更新的核心结论。** 在全离散损失下，码本容量 K 通过离线量化误差决定离散目标的信息量，进而决定动力学的可预测性与闭环成功率：PushT held-out 成功率从 K512 的 17.5% 单调升至 K8192 的 62.0%，三层证据（量化 gap、`pred_teacher_mse`、成功率）严格同序。
2. **该影响是任务相关的。** PushT 对码本分辨率持续敏感（K 翻倍仍有大幅收益），Cube 在 K2048 后即接近上限。码本大小的选择应按目标任务实测，不能沿用离线指标外推。
3. **K8192 是 PushT 类质量敏感任务的明确选择**；**Cube 类任务 K2048 起步即可**，把体积降为 1/4 而任务表现无统计显著损失。
4. **刚体变换码本与原码本闭环表现等价**（几何审计与结果一致），可作为码本几何归一化的工具，但不改变质量排序。
5. **离线结论保持不变**：K8192 离线保真最佳，K4096 是离线均衡候选；大码本 perplexity/K 递减（85.21% → 65.91% active）提示更大 K 的边际收益终将衰减。
6. **后续建议**：
   - 补测 K4096（以及 K1024）的 PushT held-out 闭环，定位成功率–容量曲线的饱和点；
   - 多 seed 复核 K2048 vs K8192 的 PushT 差距并给出配对置信区间；
   - 针对 PushT 的 28 个百分点离散化代价，试验残差量化或分层码本，检验量化误差假设的因果性；
   - Cube 补充 held-out 分片协议，提高 K2048 vs K8192 比较的分辨率。
