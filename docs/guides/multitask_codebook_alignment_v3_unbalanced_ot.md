# PushT 与 Two-Room 隐空间对齐及码本融合方案

## 1. 目标

本实验希望从 PushT 和 Two-Room 的官方 LeWM 模型中分别提取连续隐表示并训练独立码本，随后将两套隐空间和码本对齐到同一个公共空间，最终训练一个能够同时完成两个任务的模型。

最终部署模型应尽量只包含：

- 一个共享视觉编码器；
- 一个共享离散码本；
- 一个共享动力学 predictor；
- 一个共享动作编码器；
- 一个很小的 task embedding，用于区分动作语义和任务动力学。

两个官方教师、隐空间对齐器以及教师侧辅助模块只在离线缓存和蒸馏训练期间使用，不进入最终部署模型。

评估矩阵中另设一个原生连续多任务模型 M3 作为 baseline。M3 不使用教师隐空间对齐、码本融合或蒸馏；它直接使用官方 PushT 与 Two-Room 数据，按原版 LeWM 训练方法从头联合训练。该模型用于判断“对齐与码本融合”相对于“直接进行原生多任务训练”是否带来收益，不作为上述离散部署形态的一部分。

## 2. 第二任务选择

推荐使用 **Two-Room**：

- 官方模型：[quentinll/lewm-tworooms](https://huggingface.co/quentinll/lewm-tworooms)
- 官方数据：[quentinll/lewm-tworooms](https://huggingface.co/datasets/quentinll/lewm-tworooms)
- 环境：`swm/TwoRoom-v1`
- 本仓库数据配置：`scripts/train/config/data/tworoom.yaml`
- 本仓库规划配置：`scripts/plan/config/tworoom.yaml`

PushT 和 Two-Room 的官方模型均使用 ViT-Tiny、224×224 输入、192 维 latent、3 帧历史和相同规格的 predictor，动作编码器输入也均为 10 维。因此，两者比 Cube 更适合先验证双任务码本融合。

## 3. 为什么不能直接拼接或平均两个码本

设两个冻结教师编码器为

$$
z_P=f_P(x),\qquad z_R=f_R(x),\qquad z_P,z_R\in\mathbb{R}^{192},
$$

其中下标 $P$ 表示 PushT，$R$ 表示 Two-Room。

即使两个模型结构完全相同，独立训练也允许隐空间发生旋转、反射、平移、缩放甚至非线性扭曲。例如，对任意正交矩阵 $Q$，只要同步修改 predictor，$z$ 和 $zQ$ 可以取得相同的预测损失和任务性能。

因此：

- 两个码本中相同的 token index 没有天然对应关系；
- `C_push[i]` 与 `C_room[i]` 不能直接平均；
- 两个空间中的欧氏距离不能在未经对齐时直接比较；
- 仅仅因为 latent 维度都是 192，并不意味着它们处于同一个坐标系。

正确顺序是：

1. 选择公共隐空间；
2. 对齐两个教师空间；
3. 在公共空间中重新构造融合码本；
4. 用融合码本蒸馏共享学生模型。

## 4. 总体架构

```text
PushT 图像 ──> 冻结 PushT 教师 ──> z_push ──> A_push ──┐
                                                       ├──> 公共教师空间 u
TwoRoom 图像 -> 冻结 TwoRoom 教师 -> z_room -> A_room ─┘
                                                                 │
                                                                 v
                                                        融合码本 C_shared

PushT / TwoRoom 图像 ──> 共享学生编码器 E ──> student latent ──> C_shared
                                                                 │
动作 + task embedding ───────────────────────────────────────────>│
                                                                 v
                                                       共享 predictor F
```

第一版以 PushT 教师空间作为参考空间：

$$
A_P(z)=z,
$$

并学习 Two-Room 到 PushT 参考坐标系的映射 $A_R$。

## 5. 阶段一：分别建立单任务基线

在融合前必须完成以下四个独立基线：

1. 官方连续 PushT LeWM；
2. PushT 单任务 VQ-LeWM；
3. 官方连续 Two-Room LeWM；
4. Two-Room 单任务 VQ-LeWM。

两套码本建议先保持相同规格：

```yaml
codebook:
  num_embeddings: 8192
  embedding_dim: 192
```

需要记录：

- train/test quantization MSE；
- 相对量化误差；
- perplexity；
- dead-code 比例；
- token agreement；
- one-step prediction loss；
- rollout prediction gap；
- 最终任务成功率。

只有两个单任务 VQ 模型都接近各自连续教师时，融合实验才具有可解释性。

## 6. 阶段二：构造跨教师校准集

### 6.1 同图像 anchor

从两个任务各抽取等量图像：

```text
X_cal = 50% PushT 图像 + 50% Two-Room 图像
```

将同一张图像同时输入两个冻结编码器：

```python
z_push = push_teacher.encode({'pixels': images})['emb']
z_room = room_teacher.encode({'pixels': images})['emb']
```

由此得到成对表示：

$$
\left\{\left(z_{R,i},z_{P,i}\right)\right\}_{i=1}^{N}.
$$

校准集应再划分为 alignment train 和 held-out alignment validation。不能在用于拟合映射的同一批 latent 上报告对齐质量。

### 6.2 为什么使用两个任务的图像

如果只用 PushT 图像拟合，Two-Room 编码器始终处于分布外输入；只用 Two-Room 图像时，PushT 编码器也存在同样问题。使用 50/50 混合图像能够让映射同时覆盖两个模型的原生输入分布。

## 7. 阶段三：Similarity Procrustes 对齐

第一版使用旋转/反射、全局缩放和平移：

$$
\min_{s,R,b}
\left\|sZ_RR+b-Z_P\right\|_F^2,
\qquad R^\top R=I.
$$

相比任意线性层，这种变换只对欧氏距离做统一缩放，能够保持 Two-Room 原码本内部的最近邻关系。

参考实现：

```python
@torch.no_grad()
def fit_similarity_procrustes(source, reference):
    source = source.float()
    reference = reference.float()

    source_mean = source.mean(dim=0, keepdim=True)
    reference_mean = reference.mean(dim=0, keepdim=True)
    source_centered = source - source_mean
    reference_centered = reference - reference_mean

    u, singular_values, vh = torch.linalg.svd(
        source_centered.T @ reference_centered
    )
    rotation = u @ vh
    scale = singular_values.sum() / source_centered.square().sum().clamp_min(1e-12)
    bias = reference_mean - scale * source_mean @ rotation
    return rotation, scale, bias


def apply_alignment(latent, rotation, scale, bias):
    return scale * latent @ rotation + bias
```

对 Two-Room latent 和码本使用完全相同的变换：

```python
aligned_room_latent = apply_alignment(room_latent, rotation, scale, bias)
aligned_room_codebook = apply_alignment(room_codebook, rotation, scale, bias)
```

若保存为 `torch.nn.Linear(192, 192)`：

```python
aligner.weight.copy_(scale * rotation.T)
aligner.bias.copy_(bias.squeeze(0))
```

建议将以下信息保存到 `alignment.pt`：

```python
{
    'source_task': 'tworoom',
    'reference_task': 'pusht',
    'rotation': rotation.cpu(),
    'scale': scale.cpu(),
    'bias': bias.cpu(),
    'source_teacher_sha256': ...,
    'reference_teacher_sha256': ...,
    'calibration_metadata': ...,
}
```

## 8. 对齐验证

### 8.1 原任务 token 不变性

在融合前，Two-Room latent 和 Two-Room 码本经过同一个 similarity transform 后，token assignment 应基本完全不变：

```python
old_tokens = nearest_code_indices(room_latent, room_codebook)
new_tokens = nearest_code_indices(
    aligned_room_latent,
    aligned_room_codebook,
)
token_preservation = (old_tokens == new_tokens).float().mean()
```

该指标应接近 100%。否则说明变换实现、矩阵方向、数值精度或预处理存在错误。

### 8.2 Held-out 对齐质量

在 held-out anchor 上报告：

- normalized RMSE；
- $R^2$；
- cosine similarity；
- SVCCA/PWCCA；
- 映射前后的协方差谱和 effective rank；
- 线性映射相对于恒等映射的改进。

表示相似度只用于诊断，不能替代任务评估。即使两个空间能被线性 adapter 功能性连接，也不代表它们编码了完全相同的信息。

### 8.3 动力学验证

将 Two-Room 官方 predictor 包装到对齐空间：

$$
F'_R(u,a)=A_R\left(F_R\left(A_R^{-1}(u),a\right)\right).
$$

比较变换前后的 one-step prediction error。如果 similarity transform 和逆变换实现正确，误差应主要来自浮点数值误差。

## 9. 阶段四：码本融合

### 9.1 实验 A：零合并拼接

首先构造容量为 16384 的码本：

$$
C_{16384}=\left[C_P;A_R(C_R)\right].
$$

该实验回答：在不牺牲容量的情况下，一个共享模型能否同时使用两套 token 完成两个任务。

这不是最终方案，但它是非常重要的上界和故障定位基线。如果 `K=16384` 都失败，问题更可能来自共享编码器、共享 predictor、任务条件或训练采样，而不是跨任务 code 的匹配与合并。

### 9.2 实验 B：基于 Unbalanced OT 的自适应 code 合并

融合算法不预先指定最终词表大小，也不对全部 latent 重新执行固定 $K$ 的 k-means。它从对齐后的两套原始码本出发，使用非平衡最优传输（Unbalanced Optimal Transport, UOT）估计跨任务 code 的软对应关系，再只合并具有足够证据表明彼此接近的 code：

$$
C_0=\left[C_P;A_R(C_R)\right].
$$

设两个单任务码本大小分别为 $K_P$ 和 $K_R$，最终接受的跨任务一对一合并集合为 $\mathcal{M}$，则：

$$
K_{\mathrm{shared}}=K_P+K_R-|\mathcal{M}|.
$$

当两边均为 8192 个 code 时，最终 $K_{\mathrm{shared}}$ 自适应地落在 $[8192,16384]$ 内，而不是被强制设为 8192。

#### 9.2.1 统计每个 code 的真实支持

对 PushT code $i$ 和 Two-Room code $j$ 分别记录：

- assignment count $n_i^P,n_j^R$；
- 任务内归一化质量 $a_i=n_i^P/N_P$、$b_j=n_j^R/N_R$；
- 被分配 latent 的平均平方量化误差 $e_i^P,e_j^R$；
- 归一化量化失真贡献 $S_i^P=a_ie_i^P$、$S_j^R=b_je_j^R$；
- RMS 或 95% 分位聚类半径 $r_i^P,r_j^R$；
- 最近的同任务 code 距离；
- code 在 held-out 数据上的使用频率。

两个任务用于统计这些量的 latent 样本数必须相同。若原始数据量不同，先在每个任务内使用固定随机种子等量采样。这样每个任务的总质量相同，传输结果不会被数据集大小支配。

没有真实 assignment 的 dead code 不应被人为赋予均匀质量。第一阶段将其排除在 UOT 匹配之外，并作为未匹配 code 保留或单独标记。

#### 9.2.2 构造 UOT 分布与传输代价

将两个对齐后的码本写成离散分布：

$$
\mu_P=\sum_{i=1}^{K_P}a_i\delta_{c_i^P},
\qquad
\mu_R=\sum_{j=1}^{K_R}b_j\delta_{A_R(c_j^R)}.
$$

基础传输代价采用对齐空间中的平方欧氏距离：

$$
D_{ij}=\left\|c_i^P-A_R(c_j^R)\right\|_2,
\qquad
C_{ij}=D_{ij}^2.
$$

若不同 latent 区域的局部尺度差异明显，可将局部尺度作为消融项：

$$
C_{ij}^{\mathrm{local}} =
\frac{D_{ij}^2}
{\left(r_i^P+r_j^R\right)^2+\epsilon_r}.
$$

主实验应固定一种代价定义，并始终额外报告原始欧氏距离，避免局部归一化掩盖绝对距离很大的错误匹配。还可以预先使用宽松的距离门 $D_{ij}\leq\tau_{\mathrm{candidate}}$ 移除明显不可能匹配的边，以降低计算量并防止远距离质量泄漏；该门不能用于强制产生指定数量的候选。

#### 9.2.3 求解 Unbalanced OT

计算非负传输矩阵 $\Pi\in\mathbb{R}_+^{K_P\times K_R}$：

$$
\Pi^* =
\arg\min_{\Pi\geq0}
\left[
\langle\Pi,C\rangle
+\varepsilon_{\mathrm{OT}}\sum_{ij}\Pi_{ij}\left(\log\Pi_{ij}-1\right)
+\rho_P\operatorname{KL}\left(\Pi\mathbf{1}\,\|\,a\right)
+\rho_R\operatorname{KL}\left(\Pi^\top\mathbf{1}\,\|\,b\right)
\right].
$$

其中：

- $\varepsilon_{\mathrm{OT}}$ 控制熵正则强度和传输矩阵的平滑程度；
- $\rho_P,\rho_R$ 控制传输边缘偏离两个码本原始质量分布的代价；
- 较小的 $\rho$ 允许任务专属或离群 code 的质量保持未匹配；
- 较大的 $\rho$ 更接近平衡 OT，可能迫使不相似 code 建立对应。

这里不能使用要求边缘质量严格相等的 balanced OT 作为主方法。两个任务合理地包含专属 code，因此算法必须允许一部分质量不参与传输。UOT 输出的 $\Pi^*$ 只是软对应矩阵，不是最终融合码本；也不能把所有非零传输边都直接视为合并关系。

#### 9.2.4 从软传输矩阵提取高置信合并对

熵正则通常会使 $\Pi^*$ 稠密。为区分“传输了多少质量”和“传输是否集中”，对每个候选对计算：

$$
q_i^{\mathrm{keep}} =
\frac{\sum_l\Pi^*_{il}}{a_i+\epsilon},
\qquad
q_j^{\mathrm{keep}} =
\frac{\sum_k\Pi^*_{kj}}{b_j+\epsilon},
$$

以及双向条件传输占比：

$$
p_{j\mid i} =
\frac{\Pi^*_{ij}}{\sum_l\Pi^*_{il}+\epsilon},
\qquad
p_{i\mid j} =
\frac{\Pi^*_{ij}}{\sum_k\Pi^*_{kj}+\epsilon}.
$$

第一阶段采用保守的一对一硬化规则。候选对 $(i,j)$ 必须同时满足：

1. $j=\arg\max_l\Pi^*_{il}$，且 $i=\arg\max_k\Pi^*_{kj}$，即双向最大传输匹配；
2. $\min\left(q_i^{\mathrm{keep}},q_j^{\mathrm{keep}}\right)\geq\tau_{\mathrm{keep}}$，两个 code 都有足够质量真正参与传输；
3. $\min\left(p_{j\mid i},p_{i\mid j}\right)\geq\tau_{\mathrm{mass}}$，传输关系在两个方向上都足够集中；
4. $D_{ij}\leq\tau_r\left(r_i^P+r_j^R\right)$，两个 code 在局部尺度上足够接近；
5. 合并代价不超过 validation split 确定的量化质量阈值。

双向最大约束使每个原始 code 至多参与一次合并，防止一个高频 code 吞并多个任务专属 code。后续若实验表明存在稳定的一对多对应，再把硬化步骤扩展为容量受限的 bipartite matching；第一阶段不采用一对多融合。

用于质量门控的 Ward 合并代价为：

$$
\Delta_{ij} =
\frac{a_ib_j}{a_i+b_j}
\left\|c_i^P-A_R(c_j^R)\right\|_2^2,
$$

其相对于两个 code 原始量化误差的归一化代价为：

$$
\gamma_{ij} =
\frac{\Delta_{ij}}
{S_i^P+S_j^R+\epsilon}.
$$

只有同时满足 UOT 双向匹配、保留质量、传输集中度、局部距离和量化质量约束的候选，才进入最终合并集合 $\mathcal{M}$。

#### 9.2.5 合并 code

接受合并后，新 code 使用任务内占用率加权中心：

$$
c_{ij}^{\mathrm{shared}} =
\frac{a_ic_i^P+b_jA_R(c_j^R)}
{a_i+b_j}.
$$

未匹配或未通过阈值的 code 原样保留。因此融合码本自然包含：

- 未匹配的 PushT 专属 code；
- 未匹配的 Two-Room 专属 code；
- 由高置信 UOT 匹配产生的共享 code。

整个过程不移动未合并 code，也不为了达到指定词表大小而合并远距离 code。

### 9.3 UOT 超参数与自适应停止条件

UOT 需要决定允许多少质量保持未匹配，以及多强的软对应关系才足以合并。相关参数应控制匹配置信度和量化质量，而不是直接控制最终 code 数量。推荐在 alignment validation split 上扫描：

- 质量松弛系数 $\rho_P,\rho_R$，第一阶段令二者相等；
- 熵正则 $\varepsilon_{\mathrm{OT}}$；
- code 保留质量阈值 $\tau_{\mathrm{keep}}$；
- 双向条件质量阈值 $\tau_{\mathrm{mass}}$；
- 局部距离阈值 $\tau_r$；
- 归一化 Ward 代价阈值 $\gamma$。

传输代价的整体尺度会影响 $\rho$ 和 $\varepsilon_{\mathrm{OT}}$ 的含义，因此参数扫描前需要固定 cost normalization，并将其写入实验 metadata。

不以“传输矩阵是否收敛”作为 code 合并的停止条件，也不根据非零传输边数量决定最终词表。对每组 UOT 参数和硬化阈值构造候选融合码本，然后在两个任务的真实 held-out latent 上重新计算 quantization error。

选择满足以下条件时合并数量最多的参数组合：

$$
\frac{QE_P(C_{\mathrm{shared}})}{QE_P(C_P)}\leq1+\varepsilon_P,
\qquad
\frac{QE_R(C_{\mathrm{shared}})}{QE_R(A_R(C_R))}\leq1+\varepsilon_R.
$$

其中 $\varepsilon_P$ 和 $\varepsilon_R$ 是预先设定的最大量化误差退化预算。若多个参数组合得到相同合并数，优先选择合并集合跨随机种子更稳定、最坏任务量化误差更低的组合。

阈值和 UOT 超参数只能使用 validation split 选择。最终结果必须在独立 test split 上报告，不能使用 test split 继续调整参数。

最终结果应同时报告：

- 自适应得到的 $K_{\mathrm{shared}}$；
- UOT 的 $\rho_P,\rho_R,\varepsilon_{\mathrm{OT}}$、cost normalization 和候选距离门；
- 总传输质量及两侧保留/丢弃的边缘质量比例；
- 双向最大匹配数和实际合并数；
- 合并比例；
- code 保留质量和双向条件传输占比分布；
- 合并距离及归一化 Ward 代价分布；
- 两个任务各自的量化误差变化；
- 两个任务对共享 code 的实际使用率。

## 10. 阶段五：离线多任务教师缓存

当前缓存脚本在教师编码后立刻计算其到单任务码本的距离。多任务版本应改成：

```python
teacher_latent = teachers[task_id].encode({'pixels': pixels})['emb']
aligned_teacher_latent = teacher_aligners[task_id](teacher_latent)

distances = squared_distances(
    aligned_teacher_latent.reshape(-1, 192),
    fused_codebook,
)
```

缓存样本至少包含：

```python
{
    'pixels': pixels,
    'action': action,
    'task_id': task_id,
    'teacher_latent': aligned_teacher_latent,
    'hard_tokens': hard_tokens,
    'topk_indices': topk_indices,
    'topk_probs': topk_probs,
}
```

缓存 metadata 需要额外记录：

- 两个数据集名称和哈希；
- 两个教师 checkpoint 的哈希；
- 两个原始码本的哈希；
- alignment checkpoint 的哈希；
- 融合码本的哈希及自适应得到的 `K_shared`；
- UOT 的代价定义、cost normalization、`rho_P`、`rho_R`、`epsilon_OT`、距离/质量阈值和量化误差预算；
- 被接受的跨任务匹配对及新旧 token ID 映射；
- 每个原始 code 的 assignment count、量化误差和局部半径；
- 每任务样本数量；
- 每任务 train/val/test 划分索引；
- `task_id` 到任务名的映射。

训练配置中的 `num_embeddings` 必须从融合码本 checkpoint 的权重形状或 metadata 读取，不能继续硬编码为 8192。

教师和对齐器退出进程后，再启动正式训练，保持当前 teacher-free DDP 设计。

## 11. 阶段六：共享模型结构

建议的共享模型为：

```python
class MultiTaskJointDistillationLeWM(nn.Module):
    def __init__(self, ...):
        self.student_encoder = ...       # shared
        self.projector = ...             # shared
        self.student_adapter = ...       # shared
        self.codebook = ...              # shared and frozen initially
        self.action_encoder = ...        # shared
        self.task_embedding = nn.Embedding(2, 192)
        self.predictor = ...              # shared
        self.pred_proj = ...              # shared
```

动作条件：

```python
def predict(self, latent, action, task_id):
    action_embedding = self.action_encoder(action)
    task_embedding = self.task_embedding(task_id)[:, None, :]
    conditioning = action_embedding + task_embedding
    prediction = self.predictor(latent, conditioning)
    return self.project_prediction(prediction)
```

虽然两个任务的动作输入都是 10 维，但动作语义不同，因此第一版必须加入 task embedding。否则共享 predictor 需要同时从图像中推断任务身份，会将“码本融合是否成功”和“任务识别是否成功”混为一个问题。

严格的单模型要求仍然成立：task embedding 只是同一模型内部的条件输入，不是两套独立 predictor。

## 12. 联合训练目标

设当前任务为 $k$，对齐后的教师 latent 为 $u_k$，共享学生 latent 为 $s$，融合码本为 $C$。推荐损失：

$$
\mathcal{L} =
\lambda_z\mathcal{L}_{\mathrm{latent}}
+\lambda_{\mathrm{KL}}\mathcal{L}_{\mathrm{token}}
+\lambda_{\mathrm{dyn}}\mathcal{L}_{\mathrm{prediction}}.
$$

其中：

$$
\mathcal{L}_{\mathrm{latent}}=\left\|s-u_k\right\|_2^2,
$$

$$
\mathcal{L}_{\mathrm{token}} =
\operatorname{KL}\left(
    p_C(u_k)\,\|\,p_C(s)
\right),
$$

$$
\mathcal{L}_{\mathrm{prediction}} =
\left\|
F\left(s_{t-h:t},a_{t-h:t},k\right)-s_{t+1}
\right\|_2^2.
$$

这与当前 PushT 联合蒸馏中的三项损失一致。主要变化是：

- `teacher_latent` 已在缓存阶段映射到公共空间；
- `hard_tokens/topk` 来自融合码本；
- `predict()` 接收 `task_id`；
- batch sampler 保证两个任务采样平衡。

## 13. 训练策略

### Phase 0：固定教师空间

- 冻结两个官方教师；
- 拟合并冻结 Procrustes 对齐器；
- 构造并冻结融合码本；
- 生成完整离线缓存。

### Phase 1：学习共享编码器

- 冻结融合码本；
- 优先优化 student encoder、projector 和 student adapter；
- predictor 使用较低学习率或暂时冻结；
- 主要关注 latent MSE、token agreement 和每任务 perplexity。

### Phase 2：联合学习共享动力学

- 训练共享 encoder、action encoder、task embedding 和 predictor；
- 沿用 teacher forcing 从教师 code 向学生 latent 逐步过渡；
- 每个 optimizer step 使用相同数量的 PushT 和 Two-Room 样本。

### Phase 3：学生闭环微调

- 使用 100% student latent rollout；
- 降低 encoder 和 predictor 学习率；
- 仍冻结码本，避免 token 语义在最后阶段漂移；
- 分别在两个环境上执行自动评估。

第一轮实验不建议同时学习对齐器、码本、学生编码器和 predictor，否则很难判断失败来源，也容易出现坐标漂移或单任务垄断。

## 14. 多任务采样

不能直接拼接两个数据集后普通 shuffle，因为大数据集会支配训练。推荐每个 global batch 满足：

```text
50% PushT + 50% Two-Room
```

可使用两个独立 `DistributedSampler`，每一步分别从两个 loader 取半个 batch，再拼接：

```python
batch = concat_batches(
    next(push_loader),
    next(room_loader),
)
```

验证指标必须按任务分别汇总，不能只报告混合平均值。

## 15. 评估矩阵

至少比较以下模型：

| 编号 | 模型 | 码本 | 用途 |
|---|---|---|---|
| P0 | 官方 PushT | 连续 | PushT 教师上界 |
| P1 | 单任务 PushT VQ | K=8192 | PushT 离散基线 |
| R0 | 官方 Two-Room | 连续 | Two-Room 教师上界 |
| R1 | 单任务 Two-Room VQ | K=8192 | Two-Room 离散基线 |
| M0 | 多任务共享模型 | 未对齐、零合并 K=16384 | 负对照 |
| M1 | 多任务共享模型 | Procrustes、零合并 K=16384 | 不合并上界 |
| M2 | 多任务共享模型 | Procrustes、自适应 $K_{\mathrm{shared}}$ | 推荐融合方案 |
| M3 | 原生多任务模型 | 连续 latent，无对齐、无融合码本 | 官方原版训练方法的等参数 baseline |

M3 直接混合官方 PushT 与 Two-Room 训练数据，以任务平衡采样器训练一个原生支持两个任务的单一模型。除联合数据加载和必要的 task_id 条件外，损失函数、优化器、学习率计划和训练流程遵循官方原版 LeWM 方法；不得使用两个单任务教师的 latent、Procrustes 映射、融合码本或蒸馏 loss。

参数量公平性按以下口径控制：

- M3 与 M0/M1/M2 使用完全相同宽度和深度的视觉编码器、dynamics predictor、动作编码器、预测头和 task embedding；
- 不允许给 M3 增加任务专属 encoder、predictor、expert 或额外隐藏层；
- 四个模型的可训练网络参数量必须一致，并在报告中给出逐模块参数统计；
- M0/M1/M2 的冻结码本存储量单独报告，不计入可训练参数量。由于 M0/M1 为 K=16384、M2 为自适应 $K_{\mathrm{shared}}$，若把冻结码本也计入总参数，三者本身就不可能具有完全相同的总张量数。

所有多任务模型报告：

- PushT success rate；
- Two-Room success rate；
- 两任务相对各自单任务 VQ 基线的性能下降；
- 总参数量、可训练参数量和逐模块参数量；
- 每任务训练 loss、验证 loss、吞吐和峰值显存。

M0/M1/M2 额外报告：

- 自适应得到的最终 $K_{\mathrm{shared}}$、候选数、合并数和合并比例；
- 每任务合并前后的 quantization MSE、相对误差及误差增量；
- 合并距离、局部半径比和归一化 Ward 代价分布；
- 每任务 perplexity/dead-code；
- 每任务 token agreement；
- 每任务 rollout prediction gap；
- code usage overlap 和共享 code 实际使用率；
- token 与 task ID 的互信息 $I(\mathrm{token};\mathrm{task})$。

M3 额外报告原版连续 latent 训练目标的分任务指标，并分别计算它相对 P0/R0 连续教师和 M0/M1/M2 的任务性能差。码本相关指标不适用于 M3。

$I(\mathrm{token};\mathrm{task})$ 很高表示码本倾向于分成两个任务分区；很低表示 token 使用更共享。但不能单独追求低互信息，因为 predictor 可能确实需要保存任务特有状态。

## 16. 成功判据

建议按以下顺序设门槛：

1. 两个单任务 VQ 模型均能接近其连续教师；
2. similarity transform 保持 Two-Room 原 token assignment；
3. 对齐后的 held-out residual 显著优于未对齐结果；
4. 等参数的原生多任务 M3 能使用官方 PushT 与 Two-Room 数据完成两个任务，为融合模型提供直接训练 baseline；
5. K=16384 零合并模型能在两个任务上工作；
6. 自适应合并在两个任务上都满足预先设定的量化误差退化预算；
7. 自适应融合模型在两个任务上都接近各自单任务 VQ 基线和原生多任务 M3；
8. $K_{\mathrm{shared}}$、合并比例和任务性能在多随机种子下稳定；
9. 放宽合并阈值时，词表大小与性能呈可解释的容量—质量曲线。

最终任务成功率是决定性指标。latent MSE、SVCCA 和 token overlap 只能解释原因，不能代替控制评估。

## 17. 原生双任务 LeWM baseline

M3 使用官方 PushT 与 Two-Room 数据，从随机初始化开始训练一个原生双任务 LeWM。它不复用两个官方单任务模型的 latent 或 predictor，也不构造离散码本，训练目标和优化流程沿用官方原版方法。

为保证它只检验“直接联合训练”这一因素，必须保持以下控制变量：

- 使用与 M0/M1/M2 相同的 train/validation/test 样本划分；
- 两个任务等概率采样，单个任务的数据量差异不能改变梯度占比；
- 使用相同的视觉输入、动作输入、history 长度和 task_id 接口；
- 共享一个视觉编码器、一个 dynamics predictor 和一个动作编码器，不允许任务专属主干；
- 主干宽度、层数、隐藏维度和可训练参数量与 M0/M1/M2 完全一致；
- 对齐全局 batch、优化步数、学习率计划、AdamW 参数、梯度裁剪、精度和训练 seed；
- checkpoint 选择和 PushT/Two-Room held-out 评测协议与 M0/M1/M2 相同。

该 baseline 的解释方式是：如果 M2 在更紧凑的离散码本下接近或超过 M3，说明教师对齐与自适应码本融合具有实际价值；如果 M3 明显优于 M2，则需要进一步区分差距来自量化、Procrustes 残差、code 合并还是蒸馏训练，而不能直接把差距归因于多任务共享本身。

## 18. 当前仓库的建议改造点

建议新增：

```text
scripts/train/fit_latent_alignment.py
scripts/train/build_fused_codebook.py
scripts/train/cache_multitask_distillation.py
scripts/train/multitask_vq_lewm_distillation.py
scripts/train/evaluate_multitask_distillation.py
scripts/train/multitask_lewm_baseline.py
scripts/train/config/multitask_vq_lewm.yaml
scripts/train/config/multitask_lewm_baseline.yaml
stable_worldmodel/wm/vq_lewm/alignment.py
stable_worldmodel/wm/vq_lewm/multitask.py
tests/wm/test_latent_alignment.py
tests/wm/test_fused_codebook.py
tests/wm/test_multitask_vq_distillation.py
tests/wm/test_multitask_lewm_baseline.py
```

现有代码对应关系：

- `scripts/train/latent_codebook.py`：复用单任务 latent 提取和码本训练；
- `stable_worldmodel/wm/latent_codebook.py`：复用 EMA teacher/student codebook；
- `scripts/train/cache_codebook_distillation.py`：扩展为双教师、对齐后缓存；
- `stable_worldmodel/wm/vq_lewm/distillation.py`：扩展共享模型并加入 task embedding；
- `scripts/train/vq_lewm_joint_distillation.py`：扩展双数据集 balanced loader 和分任务指标；
- `scripts/train/evaluate_joint_distillation.py`：扩展 PushT/Two-Room 双任务评估。

需要特别区分：当前 `JointDistillationLeWM.adapter` 是学生编码器输出后的 adapter；新增的 `teacher_aligners` 是教师空间到公共空间的离线映射。两者不应复用同一个模块或 checkpoint key。

## 19. 推荐执行顺序

```text
1. 下载并验证 Two-Room 官方模型和数据
2. 评估 Two-Room 官方连续模型
3. 使用官方 PushT + Two-Room 数据，按原版方法从头训练并评估等参数 M3 原生多任务 baseline
4. 提取 Two-Room latent 并训练 K=8192 单任务码本
5. 训练并评估 Two-Room 单任务 VQ-LeWM
6. 构造双教师同图像校准集
7. 拟合并验证 Similarity Procrustes
8. 生成对齐后的 Two-Room latent/codebook
9. 构造 K=16384 零合并拼接码本
10. 训练 K=16384 双任务共享模型，建立不合并上界
11. 统计每个 code 的 occupancy、量化误差和局部半径
12. 求解 Unbalanced OT，并从软传输矩阵提取高置信一对一候选合并对
13. 在 validation split 上联合选择 UOT 参数和合并阈值，按量化质量预算得到自适应 K_shared
14. 使用自适应融合码本训练双任务共享模型
15. 完成 M3 baseline 对照、阈值曲线、消融、多种子训练和双环境评估
```

## 20. 核心结论

本实验中真正需要对齐的不是两个 token index，而是两个教师 latent 的几何和功能坐标系。Similarity Procrustes 是成本最低、容易验证并且保持原码本最近邻结构的第一步；融合阶段不固定最终词表大小，只在对齐空间中合并高置信度的近邻 code，未匹配 code 原样保留。

原生多任务 M3 不依赖教师空间对齐或码本融合，它提供了同等可训练参数预算下“直接联合训练”的必要 baseline。M2 与 M3 的差异用于判断对齐、蒸馏和自适应离散码本是否优于直接使用官方数据训练连续多任务模型；最终结论必须同时报告两个任务的 held-out 表现，不能只比较混合平均值。

## 参考资料

- [LeWorldModel 官方仓库](https://github.com/lucas-maes/le-wm)
- [PushT 官方 LeWM checkpoint](https://huggingface.co/quentinll/lewm-pusht)
- [Two-Room 官方 LeWM checkpoint](https://huggingface.co/quentinll/lewm-tworooms)
- [SVCCA: Singular Vector Canonical Correlation Analysis](https://proceedings.neurips.cc/paper/2017/hash/dc6a7e655d7e5840e66733e9ee67cc69-Abstract.html)
- [Bridging Large Gaps in Neural Network Representations with Model Stitching](https://proceedings.mlr.press/v322/traft26a.html)
- [Functional Alignment Can Mislead: Examining Model Stitching](https://proceedings.mlr.press/v267/smith25a.html)
