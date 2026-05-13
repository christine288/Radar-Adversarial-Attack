# 雷达航迹分类（MAT 数据 + Transformer）

## 项目介绍

用 **Transformer 编码器（PyTorch 开源 `nn.TransformerEncoder`）** 对**雷达目标航迹**做**多类别分类**。  
数据**仅支持 MATLAB v7.3（HDF5）** 的 `.mat` 文件，由 `mat_loader.py` 读入；每条 `.mat` 为一条**时序航迹**。

**每条航迹在代码中的形式**：序列 **`(T, C)`**，其中 **`C = 12`**（主序列列顺序见下表），按时间排序。训练/验证/测试均使用**同一套 12 列**经 `data_utils.preprocess_track_channels` 预处理后，直接送入 `model.RadarTrackTransformer`（输入形状 `(B,T,C)`）。

| 列名（HDF5 根数据集） | 含义（与 DAUR TR 说明一致） |
|------------------------|-----------------------------|
| `V_m` | 速度测量（m/s） |
| `R_m` | 距离测量（若数值为 km 量级，加载时自动换算为米） |
| `A_m` | 方位角测量（°） |
| `V` / `R` / `A` | 速度 / 距离 / 方位关联值 |
| `E_m` / `E` | 俯仰测量 / 俯仰关联 |
| `DATA_time` | 北京时间（当天累计秒等，按原始单位读入） |
| `GPS_time_in_data` | GPS 时间（毫秒累计等） |
| `Iframecnt` | 帧计数 |
| `SNR` | 信噪比 |

列顺序由 `data_utils.TRACK_CHANNEL_KEYS` 定义，**MAT 中须能读到上述 12 个根级 Dataset**；若缺字段会报错。  
常量 `NUM_TRACK_CHANNELS = 12`。

**预处理与输入 Transformer**：

1. `mat_loader.load_mat_track` / `load_mat_directory` 读入并对齐长度；**`R_m` 与 `R`** 共用「若最大距离 &lt; 100 则视为 km」规则，乘以 `range_scale`（默认 1000）转为米。  
2. `data_utils.preprocess_track_channels`：对多通道分别做对数/归一化等（前两组 `V_m,R_m,A_m` 与 `V,R,A` 使用 `log_preprocess` 思路；俯仰、时间、帧号、SNR 有各自规则）。  
3. `batch_tracks_to_sequences`：将每条预处理后轨迹堆叠为 `X.shape=(B,T,C)`，标签为 `y.shape=(B,)`。  
4. `RadarTrackTransformer` 内部流程：`Linear(C->d_model)` + 位置编码 + `TransformerEncoder` + 时序平均池化 + 分类头。

**调试**：加载后可在控制台打印前几行表格（需 pandas）：

- 单文件：`load_mat_track(..., show_preview=True)`，或调用 `mat_loader.inspect_track_sample(sample, ...)`  
- 训练：命令中必须带 **`--mat_show_preview`**，否则无预览；终端里搜索 **`MAT track 预览`** 横幅（仅打印**第一条**成功加载的样本）

**导出 MAT 为表格文件**：`analyze_mat_to_table.py`（生成 UTF-8 CSV，含轨迹表与可选 `File_head` 元数据）。

---

## 目录结构（训练 / 测试必须分开）

至少 **2 个类别**，每个类别一个子文件夹：

```text
project_data/
  mat_train/                 # 只给 train.py（内部再划分训练 / 验证）
    类别A/
      *.mat
    类别B/
      *.mat
  mat_test/                  # 给 evaluate.py / robust_evaluate.py / whitebox_attacks.py / blackbox_attacks.py 等
    类别A/
      *.mat
    类别B/
      *.mat
```

**不要把测试用 `.mat` 放进 `mat_train/`。**  
`whitebox_attacks.py`、`blackbox_attacks.py` 与 `evaluate.py` 一样使用**测试目录**（`--mat_test_dir` / `--mat_dir`），并从与权重同名的 **`*.pth.meta.json`** 读取 `mat_target_points`（若存在），否则默认重采样长度 **32**。

---

## 环境准备

```bash
cd D:\code\python\radarTransformer
pip install -r requirements.txt
```

依赖：**PyTorch**、**numpy**、**pandas**、**h5py**（读 v7.3 `.mat`）等，见 `requirements.txt`。

---

## GPU

`train.py`、`evaluate.py`、`robust_evaluate.py`、`whitebox_attacks.py`、`blackbox_attacks.py` 默认 **`--device cuda`**；无 CUDA 时回退 CPU 并警告。  
强制必须有 GPU：`--require_gpu`。只用 CPU：`--device cpu`。

---

## 训练

```bash
python train.py --mat_train_dir D:\code\python\radarTransformer\project_data\mat_train --save_path model_transformer.pth --mat_val_ratio 0.3 --split_seed 123 --batch_size 8 --epochs 100 --lr 0.001
```

要在终端看到首条航迹的数值表：

```bash
python train.py --mat_train_dir D:\code\python\radarTransformer\project_data\mat_train --save_path model_transformer.pth --mat_show_preview
```

预览出现在 **`[data] total=...` 之前**。请在本机 **终端 / 命令行**里看（VS Code / Cursor 请打开 **终端** 面板，不要用仅显示「问题」的窗口）。输出里会包含横幅行 **`========== MAT track 预览`**，其下为 pandas 打印的若干行 `track` 列。

`train.py` **默认超参**（可用 `--help` 查看全部）：`--epochs 100`、`--batch_size 200`、`--lr 1e-5`、`--dropout 0.5`、`--mat_target_points 32`、`--mat_val_ratio 0.2`、`--split_seed 42`。下文示例中的学习率 / batch 等为**可调示例**，不必与默认一致。

常用参数：

| 参数 | 说明 |
|------|------|
| `--mat_train_dir` / `--mat_dir` | 训练+验证用 `.mat` 根目录（二选一，优先前者） |
| `--mat_target_points` | 每条航迹重采样长度，默认 32；Transformer 输入序列长度即该值 |
| `--mat_val_ratio` | 从训练目录中划分验证集比例，默认 0.2；划分**按类别分层**，并保证训练集含每个类别至少 1 条（样本足够时） |
| `--split_seed` | 训练/验证划分随机种子 |
| `--mat_show_preview` | **可选**；为真时在加载完成后于终端打印**目录中第一条成功加载**样本的 `track` 表（调试用，需 pandas） |
| `--max_train_samples` | **调试**：加载后只保留前 `N` 条再划分 train/val（见下文「少量样本过拟合自检」） |
| `--round_size` | 将训练过程按 `N` 个 epoch 为一轮统计验证准确率均值（默认 `50`） |
| （默认开启）类别不均衡 | 训练集自动使用 **加权交叉熵**（`balanced` 类权重）+ **`WeightedRandomSampler`**；若要与旧行为一致，加 **`--no_class_balance`** |

关闭类别均衡时，与上节主命令等价示例（在末尾追加 `--no_class_balance` 即可）：

```bash
python train.py --mat_train_dir D:\code\python\radarTransformer\project_data\mat_train --save_path model_transformer_nobal.pth --mat_val_ratio 0.3 --split_seed 123 --batch_size 8 --epochs 100 --no_class_balance
```

成功后会生成 **`dataoutput/model_transformer.pth`** 和 **`dataoutput/model_transformer.pth.meta.json`**（类别映射、重采样点数、`input_size=[T,C]`、模型类型、是否启用类别均衡及各类权重等）。

### 数据不足时：生成合成 MAT 航迹（可选）

当 `project_data/mat_train` 样本较少时，可先基于现有各类样本分布生成一批同结构 `.mat`（字段仍为 12 列：`V_m...SNR`）。新版本支持：

- 同类双样本混合 + 平滑噪声
- 对数域增强（可关闭）
- 异常点注入（模拟波动干扰）
- 按“每类目标总数”自动补齐

```bash
python generate_synthetic_mat_data.py --source_root project_data/mat_train --output_root project_data/mat_train_augmented --synthetic_per_class 100 --noise_scale 0.16 --anomaly_prob 0.02 --seed 42
```

参数说明：

- `--source_root`：原始训练集根目录（按类别子目录）
- `--output_root`：增强后输出目录（默认 `project_data/mat_train_augmented`，不覆盖原目录）
- `--synthetic_per_class`：每个类别新增多少条合成样本（默认模式）
- `--target_count_per_class`：每类目标总数（>0 时优先按总数补齐）
- `--noise_scale`：扰动强度（建议 `0.05~0.35`）
- `--disable_paper_log_enhance`：关闭对数域增强（默认开启）
- `--anomaly_prob`：异常点注入概率（建议 `0.00~0.08`）
- `--anomaly_scale`：异常点扰动倍率（相对该类通道标准差）
- `--seed`：随机种子，保证可复现
- `--no_copy_original`：仅输出合成样本，不复制原始样本

生成后可直接用增强目录训练：

```bash
python train.py --mat_train_dir D:\code\python\radarTransformer\project_data\mat_train_augmented --save_path model_aug_transformer.pth --epochs 100 --batch_size 8 --lr 0.001 --split_seed 123 --dropout 0.2  --no_class_balance

```

训练结束时会额外打印“按轮次统计验证准确率”。默认 `--round_size 50` 下，`--epochs 100` 会输出：

- 第 1 轮（Epoch 1-50）平均准确率
- 第 2 轮（Epoch 51-100）平均准确率
- 两轮均值再平均得到的最终轮次平均准确率

### 少量样本过拟合自检（可选）

用于检查「训练管线能否在极少样本上把 **Train Acc** 拉高」（若仍接近随机，优先查数据/标签/学习率）。要求训练划分里**至少 2 个类别**；`--max_train_samples 2` 会只取加载顺序下的**前 2 条**再划分 train/val。

```bash
python train.py --mat_train_dir D:\code\python\radarTransformer\project_data\mat_train --max_train_samples 2 --epochs 80 --lr 0.001 --mat_val_ratio 0.5 --save_path debug_overfit_transformer.pth
```

---

## 测试（准确率）

```bash
python evaluate.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/mat_test

python evaluate.py --model_path dataoutput/model_aug_transformer.pth --mat_test_dir project_data/mat_test
```

### 混淆矩阵与按类报告

在测试集上除总体指标外，打印 **混淆矩阵**（行=真实类别，列=预测类别）和 **classification_report**，便于观察是否总预测成某一类、某类 recall 是否为 0 等：

```bash
python evaluate.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/mat_test --confusion

python evaluate.py --model_path dataoutput/model_aug_transformer.pth --mat_test_dir project_data/mat_test_augmented --confusion
```

### 训练结束后打印验证集混淆矩阵

与 `evaluate --confusion` 格式一致，但用的是**最后一轮**的**验证集**预测，便于和测试集对比（是否同一类总被错成某一类等）：

```bash
python train.py --mat_train_dir D:\code\python\radarTransformer\project_data\mat_train --save_path model_transformer.pth --val_confusion --epochs 100
```

---

## 调参：可挨个尝试的命令（路径请按本机修改）

以下假定项目根为 `D:\code\python\radarTransformer`，训练数据为 `project_data\mat_train`。  
注意：`train.py` 会统一保存到 `dataoutput/`，`--save_path` 仅作为**输出文件名**使用（建议每次换名避免覆盖）。

**1）加大学习率、多训几轮（先看 Train Acc / Val Acc 是否上去）**

```bash
python train.py --mat_train_dir D:\code\python\radarTransformer\project_data\mat_train --save_path model_lr1e3.pth --lr 0.001 --epochs 150 --batch_size 8
```

```bash
python train.py --mat_train_dir D:\code\python\radarTransformer\project_data\mat_train --save_path model_lr3e4.pth --lr 0.0003 --epochs 150 --batch_size 8
```

**2）训练结束看验证集混淆矩阵（与测试集 `evaluate --confusion` 对照）**

```bash
python train.py --mat_train_dir D:\code\python\radarTransformer\project_data\mat_train --save_path model_valcm.pth --lr 0.001 --epochs 100 --val_confusion
```

**3）关掉类别加权与均衡采样（与默认策略对比）**

```bash
python train.py --mat_train_dir D:\code\python\radarTransformer\project_data\mat_train --save_path model_nobal.pth --lr 0.001 --epochs 100 --no_class_balance
```

**4）更温和的类别权重（`sqrt`，减轻「总猜某一类」）**

```bash
python train.py --mat_train_dir D:\code\python\radarTransformer\project_data\mat_train --save_path model_sqrtw.pth --lr 0.001 --epochs 100 --class_weight_mode sqrt
```

**5）测试集混淆矩阵与指标（换用你的权重文件名）**

```bash
python evaluate.py --model_path D:\code\python\radarTransformer\dataoutput\model_lr1e3.pth --mat_test_dir D:\code\python\radarTransformer\project_data\mat_test --confusion
```

**6）少量样本过拟合自检（管线是否正常）**

```bash
python train.py --mat_train_dir D:\code\python\radarTransformer\project_data\mat_train --max_train_samples 2 --epochs 80 --lr 0.001 --mat_val_ratio 0.5 --save_path debug_overfit.pth
```

**7）数据量**：长期需增加各类 `.mat` 条数；上列命令不能代替采集，只能减轻不均衡与超参问题。

| 参数 | 说明 |
|------|------|
| `--val_confusion` | 训练最后一轮结束后打印**验证集**混淆矩阵 |
| `--class_weight_mode sqrt` | 损失与采样用更温和的 `sqrt` 平衡（默认 `balanced`） |
| `--no_class_balance` | 完全关闭加权与 `WeightedRandomSampler` |

---

## 鲁棒性：自然噪声干扰实验（PyTorch）

在**测试集 MAT 目录**上评估（需同时指定模型与测试目录）：

```bash
# 四种条件一起对比：无噪声 / 高斯 / 椒盐 / 斑点（Speckle，乘性噪声）
python robust_evaluate.py --model_path dataoutput/model_aug_transformer.pth --mat_test_dir project_data/mat_test_augmented --condition all --max_rel_change 0.05 --worst_of_k 16 --sigma 0.8 --sp_prob 0.3 --speckle_sigma 0.8

# 只评估高斯噪声
python robust_evaluate.py --model_path model_transformer.pth --mat_test_dir project_data/mat_test --condition gaussian --sigma 0.8 --max_rel_change 0.05 --worst_of_k 16

# 只评估椒盐噪声
python robust_evaluate.py --model_path model_transformer.pth --mat_test_dir project_data/mat_test --condition salt_pepper --sp_prob 0.3 --max_rel_change 0.05 --worst_of_k 16

# 只评估斑点噪声（Speckle，乘性噪声）
python robust_evaluate.py --model_path model_transformer.pth --mat_test_dir project_data/mat_test --condition speckle --speckle_sigma 0.8 --max_rel_change 0.05 --worst_of_k 16
```

三类自然噪声参数说明：

- 高斯噪声：`--condition gaussian --sigma <标准差>`
- 椒盐噪声：`--condition salt_pepper --sp_prob <噪声概率>`
- 斑点噪声：`--condition speckle --speckle_sigma <乘性噪声标准差>`

实现见 `noise_perturbation.py`（`add_gaussian_noise` / `add_salt_pepper_noise` / `add_speckle_noise`）。

---

## 鲁棒性：白盒对抗攻击评估（PyTorch）

脚本：`whitebox_attacks.py`。在**测试集 MAT**（`--mat_test_dir` 或 `--mat_dir`）上评估 **FGSM**、**PGD**、**CW-L2**、**DeepFool**（`deepfool_attack`：多类迭代、**逐样本**反传，**很慢**，建议减小 `--batch_size`）。

### 路径与占位符

- **`--model_path`**：必须是本机存在的 **`.pth` 权重文件**路径，**不要**使用文档里的 `...` 占位符（脚本会拒绝并提示）。
- **`--mat_test_dir`**：同上，填真实目录。

### 扰动约束（与 `noise_perturbation.project_relative_change` 一致）

- 默认 **`--max_rel_change 0.05`**：逐元素相对 L∞ 盒 \(|x'-x| \le \alpha \cdot \max(|x|, \max(\varepsilon, \text{budget\_floor}))\)，其中 \(\alpha\) 即 `max_rel_change`。
- **`--no_rel_budget`**：不做相对投影（等价 `max_rel_change=None`），用于对比「是否约束过强导致成功率偏低」。
- **`--budget_floor`**：传入投影的 `min_scale`，默认 `0.0`；近零特征维可适当加大（见 `--help`）。
- **`--clamp_min` / `--clamp_max`**：需**成对**指定，攻击后再对输入做硬截断。

### 评估逻辑（与代码一致）

- 默认 **`--attack-clean-only`**（可用 `--no-attack-clean-only` 关闭）：只对「干净预测正确」的样本做攻击，**Interference Success Rate** 的分母为 clean-correct 子集。
- **FGSM**：单次；**PGD / CW / DeepFool**：每种做 **`--attack_restarts`** 次随机起点（FGSM 固定为 1 次）。
- 每次 restart：先跑 **untargeted**，再对 **`--targeted_topk`** 个「非真类且 logit 最高」的候选类做 **targeted**，用 `_select_stronger_attack_result` 取对当前 batch 更强的结果。

### 当前 CLI 默认（偏强攻击，见 `python whitebox_attacks.py --help`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--attack` | `all` | `all` = 依次跑 fgsm / pgd / cw / deepfool |
| `--batch_size` | `200` | DeepFool 大可改小，如 `32`～`64` |
| `--attack_loss` | `ce` | PGD/FGSM 用 `_attack_loss`；可选 `margin` / `margin_only` |
| `--pgd_steps` | `300` | PGD 迭代步数 |
| `--pgd_step_size` | `1.0` | 步长系数，乘到 `max_rel_change * scale` |
| `--pgd_momentum` | `0.9` | PGD 动量 |
| `--attack_restarts` | `10` | 非 FGSM 的随机重启次数 |
| `--targeted_topk` | `5` | 每样本尝试的 targeted 目标类个数 |
| `--fgsm_step_size` | `1.0` | FGSM 步长系数 |
| `--cw_steps` / `--cw_c` / `--cw_lr` / `--cw_confidence` | `1000` / `50` / `0.005` / `1.0` | CW-L2 |
| `--deepfool_steps` / `--deepfool_overshoot` | `50` / `0.02` | DeepFool |
| `--debug_grad` | 关 | 首个 batch 打印输入梯度统计 |

```bash
# 四种攻击依次评估（DeepFool 最慢）
python whitebox_attacks.py --model_path dataoutput/model_aug_transformer.pth --mat_test_dir project_data/mat_test_augmented --attack all

# 仅 PGD（使用当前默认强参数）
python whitebox_attacks.py --model_path dataoutput/model_aug_transformer.pth --mat_test_dir project_data/mat_test_augmented --attack pgd

# 仅 DeepFool（建议小 batch）
python whitebox_attacks.py --model_path dataoutput/model_aug_transformer.pth --mat_test_dir project_data/mat_test_augmented --attack deepfool --deepfool_steps 50 --batch_size 32

# 无相对预算对比
python whitebox_attacks.py --model_path dataoutput/model_aug_transformer.pth --mat_test_dir project_data/mat_test_augmented --attack pgd --no_rel_budget

# 较保守 PGD（接近历史默认）
python whitebox_attacks.py --model_path dataoutput/model_aug_transformer.pth --mat_test_dir project_data/mat_test_augmented --attack pgd --max_rel_change 0.05 --pgd_step_size 0.5 --pgd_steps 120 --attack_restarts 8 --pgd_momentum 0.75 --attack_loss margin --targeted_topk 3
```

### 输出关键字段

- **Clean Sample Accuracy**：攻击前（干净）样本准确率  
- **Attack Sample Accuracy**：攻击后准确率  
- **Interference Success Rate**：在「干净预测正确」的样本中，被攻击后预测改变（相对真实标签判错）的比例  
- **Mean / Max Rel Change**：相对扰动统计（无投影时可能很大，需结合 `--no_rel_budget` 理解）  
- **Restart Success Mean / Std**：各 restart 在当次攻击子 batch 上的成功率统计  

---

## 鲁棒性：黑盒对抗攻击评估（PyTorch）

脚本：`blackbox_attacks.py`。在**测试集 MAT**（`--mat_test_dir` 或 `--mat_dir`）上对**目标模型**做查询式黑盒攻击，实现三种策略（与脚本顶部说明一致）：

| `--attack` | 方法 | 说明 |
|------------|------|------|
| `square` | Square Attack | 基于 margin 分数的 L∞ 黑盒攻击，查询效率较高（**推荐首选**） |
| `nes` | NES-PGD | 用 Natural Evolution Strategies 估计梯度，再做 PGD 风格更新 |
| `transfer` | Transfer | 在**随机初始化**的替代模型上做白盒 PGD，将扰动迁移到目标模型 |
| `all` | 全部 | 依次运行 `square` → `nes` → `transfer`，分别打印一套指标 |

**路径**：`--model_path`、`--mat_test_dir` 须为**本机真实路径**；不能使用 `...` 等占位符（脚本会校验并报错）。权重文件须存在（例如 `dataoutput/model_transformer.pth`）；若你训练时用的 `--save_path` 是别的名字，这里要改成对应的 `dataoutput/xxx.pth`。  
**数据与长度**：与 `evaluate.py` 相同目录约定；从 `*.pth.meta.json` 读取 `mat_target_points`，缺省为 **32**。

### 扰动与评估逻辑（与白盒脚本对齐的部分）

- **`--max_rel_change`**：相对 L∞ 盒约束，语义与 `noise_perturbation.project_relative_change` 及 `whitebox_attacks.py` 一致；默认脚本内为 **0.20**（黑盒实验常用较松预算）。
- **`--no_rel_budget`**：不做相对投影，用于对比实验。
- **`--budget_floor`**：投影时的 `min_scale`，默认 **0.01**。
- **`--clamp_min` / `--clamp_max`**：需**成对**指定，攻击后对输入做硬截断。
- **`--attack-clean-only` / `--no-attack-clean-only`**：默认仅对「干净预测正确」的样本攻击；**攻击成功率（ISR）** 等指标的分母与 `whitebox_attacks.py` 含义一致。
- 每种攻击在 untargeted 之外，会对 **`--targeted_topk`** 个候选目标类做 targeted，并保留更强结果。**`--attack_restarts`** 仅作用于 **Square / NES** 的外层随机重启；**Transfer** 外层固定 **1** 轮，替代模型上的 PGD 重启由 **`--tr_restarts`** 控制。

### 推荐命令示例

**Shell 说明**：下面 **Bash** 代码块里行尾的 **`\`** 只在 **Git Bash / WSL / Linux / macOS** 下表示续行。**Windows PowerShell** 里 **`\` 不是续行符**，整段粘贴会把 `--model_path` 等当成非法表达式并报错。PowerShell 请用 **「Windows PowerShell」** 的一行命令，或把每行行尾改成 **反引号 `` ` ``** 再续行（最后一行不要加反引号）。**不要把「选项」和「参数值」拆开两行**（例如 `--targeted_topk` 与 `5` 必须在同一行）；若在 `--targeted_topk` 后误按了回车，会出现续行提示 **`>>`**，此时 Python 往往还没正常跑起来，**按 Ctrl+C** 退出后重输整行。命令正确时，Square 攻击会先做设备/模型加载，再根据数据量与查询预算**运行较久**才有最终指标输出。

**Square**

```bash
python blackbox_attacks.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/mat_test_augmented --attack square --max_rel_change 0.05 --sq_max_queries 5000 --attack_restarts 1 --targeted_topk 2
```

**减少查询次数2000**

```bash
python blackbox_attacks.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/mat_test_augmented --attack square --max_rel_change 0.05 --sq_max_queries 2000 --attack_restarts 1 --targeted_topk 1
```

**NES-PGD：**

```bash
python blackbox_attacks.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/mat_test_augmented --attack nes --max_rel_change 0.05 --nes_steps 100 --nes_samples 20 --nes_sigma 0.01 --nes_step_size 0.3 --attack_restarts 1 --targeted_topk 2```

**Transfer Attack：**

```bash
python blackbox_attacks.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/mat_test_augmented --attack square --max_rel_change 0.05 --sq_max_queries 2000 --attack_restarts 1 --targeted_topk 1
```

**三种黑盒方法全跑对比**

```bash
python blackbox_attacks.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/mat_test --attack all --max_rel_change 0.20 --batch_size 64
```

**Windows PowerShell（单行）**

```powershell
python blackbox_attacks.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/mat_test --attack all --max_rel_change 0.20 --batch_size 64
```

Square / NES 查询与迭代量大，**`--batch_size`** 默认 **64**（可用 `32`～`128` 按显存调整）。`--attack all` 时三种方法共用同一 loader，总耗时主要取决于 Square 与 NES。

### 常用 CLI 参数（完整列表见 `python blackbox_attacks.py --help`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--attack` | `all` | `square` / `nes` / `transfer` / `all` |
| `--batch_size` | `64` | 批大小，黑盒前向次数多，可适当减小 |
| `--max_rel_change` | `0.20` | 相对 L∞ 预算 |
| `--sq_max_queries` | `5000` | Square 每样本最大查询轮次上限 |
| `--sq_p_init` | `0.8` | Square 初始窗口占序列长度比例 |
| `--attack_restarts` | `3` | Square / NES 外层重启次数（Transfer 不用此项） |
| `--tr_restarts` | `5` | Transfer：替代模型 PGD 重启次数 |
| `--targeted_topk` | `5` | targeted 候选类数 |
| `--nes_samples` | `20` | NES 每步采样对数（查询约 ∝ `2 * nes_samples * nes_steps`） |
| `--nes_steps` | `100` | NES 迭代步数 |
| `--nes_sigma` / `--nes_step_size` / `--nes_momentum` | `0.01` / `0.3` / `0.9` | NES 噪声尺度、步长、动量 |
| `--tr_pgd_steps` / `--tr_pgd_step_size` / `--tr_momentum` | `200` / `0.3` / `0.9` | Transfer：替代模型 PGD 步数、步长系数、动量 |

### 输出指标（终端）

每种攻击结束会打印：攻击成功率（ISR）、平均查询轮次、干净/攻击后准确率、mAP、Target Recall（macro）、平均相对扰动与 L2 距离、SSIM、PSNR、Restart 成功率均值/标准差、失真度与已攻击样本占比等（与 `blackbox_attacks.py` 中 `_print_result` 一致）。

---

## 主要文件

| 文件 | 作用 |
|------|------|
| `train.py` | 训练 Transformer（仅 MAT） |
| `evaluate.py` | 测试集准确率（仅 MAT） |
| `whitebox_attacks.py` | 白盒攻击与评估（FGSM / PGD / CW-L2 / DeepFool） |
| `blackbox_attacks.py` | 黑盒攻击与评估（Square / NES-PGD / Transfer） |
| `mat_loader.py` | 读取 v7.3 `.mat` → `TrackSample`（12 列主序列） |
| `data_utils.py` | `TrackSample`、`TRACK_CHANNEL_KEYS`、多通道预处理、时序批处理 |
| `model.py` | `RadarTrackTransformer`（含位置编码 + TransformerEncoder） |
| `robust_evaluate.py` | 噪声鲁棒性评估（无噪声/高斯/椒盐/斑点） |
| `robust_evaluate_enhanced.py` | 扩展噪声（通道 dropout、相关漂移、脉冲簇等）与 worst-of-k 风格评估 |
| `noise_perturbation.py` | 相对投影、噪声与 `relative_change_stats`（白盒与鲁棒实验共用） |
| `device_utils.py` | 统一 CUDA/CPU 选择 |
| `analyze_mat_to_table.py` | 将单文件或目录下 MAT 导出为 CSV 表格 |

---

## `.mat` 字段说明

默认在 HDF5 **根目录**读取以下 **12 个**数据集（名称须一致）：  
`V_m`、`R_m`、`A_m`、`V`、`R`、`A`、`E_m`、`E`、`DATA_time`、`GPS_time_in_data`、`Iframecnt`、`SNR`。

若前三列在文件中的命名不同，可在 `mat_loader.load_mat_track(..., keys=("...", "...", "..."))` 中只改 **前三列** 对应 HDF5 名；其余 9 列名固定。

---

## 注意

- 多分类至少需要 **2 个类别文件夹**。  
- 训练与测试目录**文件不要交叉**，避免信息泄漏。  
- 类别文件夹名会写入 `meta.json`；测试集**子文件夹名称**应与训练时一致，以保证标签编号一致。  
- **`whitebox_attacks.py`**、**`blackbox_attacks.py`** 与 **`evaluate.py`** 共用同一套 MAT 目录约定；攻击前会先做干净前向，仅对子集做对抗（默认仅 clean-correct）。  
- 若曾使用**仅三通道**的旧模型权重，其 `input_size` 与当前 **12 通道** 流程不一致，需**重新训练**后再评估。
