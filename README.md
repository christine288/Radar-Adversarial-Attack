# 雷达航迹分类与对抗鲁棒性实验

本项目面向 **雷达目标航迹单标签多分类**，使用 PyTorch Transformer 对 MATLAB v7.3/HDF5 格式的 `.mat` 航迹文件进行训练、评估、自然噪声扰动、白盒攻击、黑盒攻击，并提供四个服务接口供外部系统调用。

当前主线任务不是 3D 检测，因此核心指标采用分类任务指标，例如 `accuracy`、`macro_f1`、`classification_mAP`、`attack_success_rate`。文档中的 `classification_mAP` 是多分类 one-vs-rest 平均精度，不是 3D 检测中的 IoU AP。

---

## 1. 项目能力概览

- 读取按类别目录组织的 MATLAB v7.3 `.mat` 航迹数据。
- 将每条航迹重采样为固定长度时序，输入形状为 `(B, T, C)`。
- 使用 `RadarTrackTransformer` 做多分类训练和测试。
- 支持自然噪声鲁棒性评估：`gaussian`、`salt_pepper`、`speckle`。
- 支持白盒攻击：`fgsm`、`pgd`、`cw`。
- 支持黑盒攻击：`square`、`nes`、`transfer`。
- 支持接口式调用：
  - 接口一：数据集格式检查。
  - 接口二：模型和数据集校验。
  - 接口三：生成扰动/对抗 `.mat` 数据集。
  - 接口四：返回分类鲁棒性评估结果。

---

## 2. 环境安装

建议使用独立虚拟环境：

```powershell
cd E:\github\Radar-Adversarial-Attack
pip install -r requirements.txt
```

依赖包括：

```text
torch
torchvision
numpy
scikit-learn
matplotlib
pandas
h5py
```

GPU 相关脚本默认优先使用 CUDA。若机器没有 CUDA，会按脚本逻辑回退 CPU 或提示错误。需要强制 GPU 时可使用 `--require_gpu`。

---

## 3. 目录说明

当前仓库中常见目录含义如下：

```text
Radar-Adversarial-Attack/
  original_data/          # 原始 MAT 数据备份，通常不直接覆盖修改
    mat_train/            # 原始训练集
    mat_test/             # 原始测试集

  project_data/           # 项目运行使用的数据目录
    train/                # 当前训练集，按类别子目录组织
    test/                 # 当前测试集，按类别子目录组织
    test_adv_pgd/         # 接口三生成的 PGD 对抗测试集
    test_adv_square/      # 接口三生成的 Square 黑盒对抗测试集
    test_adv_gaussian/    # 接口三生成的高斯自然噪声测试集

  dataoutput/             # 模型权重、模型配置和训练产物
    model_aug_transformer.pth
    model_aug_transformer.pth.meta.json
    model_dir/
      model.pth           # 模型权重和model_aug_transformer.pth一样，用于适配接口的格式
      model.pth.meta.json
      model.yaml

  output/                 # 分析脚本导出的表格、CSV 等中间结果
    1_track_table.csv
    1_track_table_file_head.csv

  __pycache__/            # Python 自动生成的缓存文件，可忽略
```

几个目录的使用原则：

- `original_data/`：建议当作原始数据备份，保留最初的 `mat_train`、`mat_test`，不要在攻击实验中直接覆盖。
- `project_data/`：当前项目训练、测试、攻击主要使用的工作数据目录。接口三会在这里生成 `test_adv_*` 这类扰动后数据集。
- `dataoutput/`：保存模型权重和模型元信息。接口二、接口三、接口四通常传入 `dataoutput/model_dir` 这类模型目录。
- `output/`：保存人工检查和数据分析导出的 CSV，不参与模型训练和攻击流程。

---

## 4. 数据格式

数据集根目录必须按类别组织：

```text
project_data/test/
  UAV/
    *.mat
  helicopter/
    *.mat
  passenger ship/
    *.mat
  speedboat/
    *.mat
```

每个 `.mat` 文件表示一条航迹。项目默认读取 12 个 HDF5 数据集字段，顺序由 `data_utils.TRACK_CHANNEL_KEYS` 定义：

| 字段 | 说明 |
|---|---|
| `V_m` | 速度测量 |
| `R_m` | 距离测量；加载时可按 km 到 m 自动换算 |
| `A_m` | 方位角测量 |
| `V` | 速度关联值 |
| `R` | 距离关联值 |
| `A` | 方位角关联值 |
| `E_m` | 俯仰测量 |
| `E` | 俯仰关联值 |
| `DATA_time` | 数据时间 |
| `GPS_time_in_data` | GPS 时间 |
| `Iframecnt` | 帧计数 |
| `SNR` | 信噪比 |

加载与预处理流程：

1. `mat_loader.load_mat_track` 读取 `.mat` 文件。
2. `mat_loader.load_mat_directory` 读取类别目录并生成标签。
3. `data_utils.resample_track` 将每条航迹重采样到固定长度，默认 `T=32`。
4. `data_utils.preprocess_track_channels` 对 12 个通道做对数、归一化、z-score 等预处理。
5. `data_utils.batch_tracks_to_sequences` 生成模型输入 `X.shape=(B,T,12)`。
6. `model.RadarTrackTransformer` 进行分类。

---

## 5. 训练

基本训练命令：

```powershell
python train.py --mat_train_dir project_data/mat_train --save_path model_transformer.pth --epochs 100 --batch_size 64 --lr 0.001
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--mat_train_dir` / `--mat_dir` | 无 | 训练数据根目录 |
| `--save_path` | `radar_transformer.pth` | 保存到 `dataoutput/` 下的权重文件名 |
| `--epochs` | `100` | 训练轮数 |
| `--batch_size` | `200` | 批大小 |
| `--lr` | `1e-5` | 学习率 |
| `--dropout` | `0.5` | dropout |
| `--mat_target_points` | `32` | 每条航迹重采样长度 |
| `--mat_val_ratio` | `0.2` | 从训练目录中划分验证集比例 |
| `--split_seed` | `42` | 训练/验证划分随机种子 |
| `--mat_show_preview` | 关闭 | 打印首条成功加载航迹的表格预览 |
| `--val_confusion` | 关闭 | 训练结束后打印验证集混淆矩阵 |
| `--no_class_balance` | 关闭 | 关闭类别加权损失和加权采样 |
| `--class_weight_mode` | `balanced` | 类别权重策略，可用 `balanced`、`sqrt` 等 |

训练成功后会生成：

```text
dataoutput/model_transformer.pth
dataoutput/model_transformer.pth.meta.json
```

其中 `.meta.json` 会记录类别映射、输入长度、输入通道数、模型结构参数等信息。评估和攻击脚本会优先读取该文件中的 `mat_target_points` 和类别信息。

---

## 6. 普通分类评估

```powershell
python evaluate.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/test
```

带混淆矩阵：

```powershell
python evaluate.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/test --confusion
```

主要输出：

- `accuracy`：总体准确率。
- `precision_macro`：宏平均精确率。
- `recall_macro`：宏平均召回率。
- `f1_macro`：宏平均 F1。
- `mAP`：多分类 one-vs-rest 平均精度。
- 混淆矩阵和 `classification_report`：用于定位某些类别是否总被错分。

---

## 7. 自然噪声鲁棒性评估

自然噪声用于模拟非优化式扰动，不属于严格对抗优化攻击。

```powershell
python robust_evaluate.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/test --condition all --max_rel_change 0.05 --worst_of_k 16
```

支持条件：

| 条件 | 说明 |
|---|---|
| `clean` | 无扰动 |
| `gaussian` | 加性高斯噪声 |
| `salt_pepper` | 椒盐噪声 |
| `speckle` | 乘性斑点噪声 |
| `all` | 依次评估全部条件 |

增强版自然扰动：

```powershell
python robust_evaluate_enhanced.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/test --condition all --max_rel_change 0.05 --worst_of_k 8
```

`robust_evaluate_enhanced.py` 额外支持脉冲簇、相关漂移、通道衰减/丢失等更结构化的自然干扰。

---

## 8. 白盒攻击

白盒攻击脚本：

```powershell
python whitebox_attacks.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/test --attack pgd --max_rel_change 0.05
```

支持方法：

| 方法 | 说明 |
|---|---|
| `fgsm` | 单步梯度符号攻击，速度快 |
| `pgd` | 多步投影梯度攻击，默认较强 |
| `cw` | CW-L2 风格攻击 |
| `all` | 依次运行三种白盒攻击 |

关键参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--max_rel_change` | `0.05` | 模型输入张量空间相对扰动预算 |
| `--budget_floor` | `0.0` | 相对预算的最小尺度地板 |
| `--fgsm_step_size` | `1.0` | FGSM 步长系数 |
| `--pgd_step_size` | `1.0` | PGD 步长系数 |
| `--pgd_steps` | `300` | PGD 迭代步数 |
| `--pgd_momentum` | `0.9` | PGD 动量 |
| `--attack_restarts` | `10` | 非 FGSM 随机重启次数 |
| `--targeted_topk` | `5` | targeted 候选目标类别数 |

---

## 9. 黑盒攻击

黑盒攻击脚本：

```powershell
python blackbox_attacks.py --model_path dataoutput/model_aug_transformer.pth --mat_test_dir project_data/test --attack square --max_rel_change 0.05 --sq_max_queries 2000 --attack_restarts 1 --targeted_topk 1
```

当前代码中，黑盒 Square 的默认核心参数已与上面命令对齐：

| 参数 | 当前默认值 |
|---|---:|
| `--max_rel_change` | `0.05` |
| `--sq_max_queries` | `2000` |
| `--attack_restarts` | `1` |
| `--targeted_topk` | `1` |
| `--budget_floor` | `0.01` |
| `--sq_p_init` | `0.8` |

支持方法：

| 方法 | 说明 |
|---|---|
| `square` | Square Attack，随机窗口块扰动，查询式黑盒攻击 |
| `nes` | NES-PGD，用 Natural Evolution Strategies 估计梯度 |
| `transfer` | 在随机替代模型上生成白盒扰动，再迁移到目标模型 |
| `all` | 依次运行三种黑盒攻击 |

### 关于 `max_rel_change`

`--max_rel_change 0.05` 约束的是 **模型输入的预处理张量空间**，不是反变换保存回 `.mat` 后的原始物理量空间。

核心约束形式为：

```python
abs(x_adv - x_clean) <= max_rel_change * max(abs(x_clean), budget_floor)
```

黑盒默认：

```python
max_rel_change = 0.05
budget_floor = 0.01
```

因此每个模型输入元素最多改动：

```python
0.05 * max(abs(x_clean), 0.01)
```

接口三 `attack_result["max_rel_change"]` 也按这个核心攻击张量空间统计，和 `blackbox_attacks.py --max_rel_change 0.05` 的口径一致。

### 黑盒输出指标

脚本会输出：

- ISR / `attack_success_rate`：干净预测正确样本中，对抗后预测错误的比例。
- 平均查询轮次与标准差。
- 干净准确率、攻击后准确率。
- 干净 mAP、攻击后 mAP。
- 平均相对扰动、最大相对扰动。
- L2 扰动、SSIM、PSNR。
- restart 成功率均值/标准差。

---

## 10. 四个接口

接口文件包括：

| 接口 | 文件 | 函数 | 作用 |
|---|---|---|---|
| 接口一 | `dataset_service.py` | `prepare_dataset(dataset_path)` | 检查数据集目录是否为 `<root>/<class_name>/*.mat` |
| 接口二 | `model_service.py` | `validate_model(model_path, dataset_path)` | 检查模型目录、权重、meta 和数据是否可用 |
| 接口三 | `attack_service.py` | `generate_adversarial(...)` | 生成自然噪声/白盒/黑盒扰动后的 `.mat` 数据集 |
| 接口四 | `eval_service.py` | `evaluate_robustness(...)` | 返回干净集与对抗集分类鲁棒性指标 |

### 调用示例：黑盒 Square

```python
from dataset_service import prepare_dataset
from model_service import validate_model
from attack_service import generate_adversarial
from eval_service import evaluate_robustness

model_path = r"dataoutput/model_dir"
dataset_path = r"project_data/test"
adv_dataset_path = r"project_data/test_adv_square"
attack_method = "square"

assert prepare_dataset(dataset_path)
assert validate_model(model_path, dataset_path)

attack_result = generate_adversarial(
    model_path=model_path,
    dataset_path=dataset_path,
    adv_dataset_path=adv_dataset_path,
    attack_method=attack_method,
)

eval_result = evaluate_robustness(
    model_path=model_path,
    dataset_path=dataset_path,
    adv_dataset_path=adv_dataset_path,
)

print("接口三 attack_result =", attack_result)
print("接口四 eval_result =", eval_result)
```

接口三会生成：

```text
project_data/test_adv_square/
  <class_name>/
    *.mat
  _interface_eval_result.json
```

`_interface_eval_result.json` 保存接口三核心攻击阶段的分类评估结果。接口四发现该文件时，会优先读取它，从而让接口四的攻击成功率口径和核心攻击代码一致，避免因 `.mat` 反变换、保存、再读取造成攻击效果被重新评估削弱。

### 接口三返回值

```python
attack_result = {
    "mean_L2": mean_L2,
    "max_L2": max_L2,
    "mad": mad,
    "mean_abs_delta": mean_abs_delta,
    "mean_rel_change": mean_rel_change,
    "max_rel_change": max_rel_change,
}
```

这些统计现在按 **核心攻击张量空间** 计算。对于 Square，`max_rel_change` 应与 `blackbox_attacks.py --max_rel_change 0.05` 口径一致。

### 接口四返回值

接口四采用分类任务指标，输出保持精简：

```python
eval_result = {
    "clean": {
        "accuracy": clean_accuracy,
        "macro_f1": clean_macro_f1,
        "classification_mAP": clean_classification_mAP,
    },
    "Adversarial": {
        "accuracy": adv_accuracy,
        "macro_f1": adv_macro_f1,
        "classification_mAP": adv_classification_mAP,
        "mAP_retention_rate": mAP_retention_rate,
        "attack_success_rate": attack_success_rate,
        "robust_accuracy": adv_accuracy,
    },
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `accuracy` | 总体分类准确率 |
| `macro_f1` | 各类别 F1 的宏平均，适合类别不均衡场景 |
| `classification_mAP` | 多分类 one-vs-rest 平均精度 |
| `mAP_retention_rate` | 对抗 mAP / 干净 mAP |
| `attack_success_rate` | 干净预测正确样本中，对抗后预测错误的比例 |
| `robust_accuracy` | 鲁棒准确率，等同对抗集准确率 |

### 接口输出样例

下面是一段黑盒 Square Attack 的实际输出示例。由于 Square Attack 本身有随机性，不同机器、不同运行次数的数值可能略有波动；这里主要用于说明运行成功后的返回格式。

```python
接口三 attack_result = {
    "mean_L2": 2669.1766168884315,
    "max_L2": 17261.566360957964,
    "mad": 2300.514473110449,
    "mean_abs_delta": 263.0509200873857,
    "mean_rel_change": 0.1735741505040966,
    "max_rel_change": 1.0002324632129562
}
```

若使用当前接口三的核心攻击张量空间统计口径重新生成，`max_rel_change` 会按核心攻击代码中的 `--max_rel_change 0.05` 口径返回；如果看到原始 `.mat` 物理量空间统计结果大于 `0.05`，请参考“重要口径说明”中的解释。

接口四返回的是分类鲁棒性指标，当前推荐格式如下：

```python
接口四 eval_result = {
    "clean": {
        "accuracy": 0.845,
        "macro_f1": 0.8421,
        "classification_mAP": 0.8797
    },
    "Adversarial": {
        "accuracy": 0.6267,
        "macro_f1": 0.6214,
        "classification_mAP": 0.7611,
        "mAP_retention_rate": 0.8652,
        "attack_success_rate": 0.2584,
        "robust_accuracy": 0.6267
    }
}
```

字段判断方式：

- `clean.accuracy` 高，说明干净测试集分类性能较好。
- `Adversarial.accuracy` / `robust_accuracy` 越高，说明模型在扰动后越稳。
- `attack_success_rate` 越高，说明攻击越有效。
- `mAP_retention_rate` 越接近 1，说明对抗扰动后排序/置信度质量保留越多。

---

## 11. 合成数据生成

当真实训练样本不足时，可以用现有样本生成同结构合成 `.mat`：

```powershell
python generate_synthetic_mat_data.py --source_root project_data/mat_train --output_root project_data/mat_train_augmented --synthetic_per_class 100 --noise_scale 0.16 --anomaly_prob 0.02 --seed 42
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--source_root` | 必填 | 原始训练集 |
| `--output_root` | 必填 | 合成数据输出目录 |
| `--synthetic_per_class` | `30` | 每类生成样本数 |
| `--target_count_per_class` | `0` | 指定每类目标总数 |
| `--noise_scale` | `0.18` | 扰动强度 |
| `--anomaly_prob` | `0.0` | 异常点注入概率 |
| `--anomaly_scale` | 见脚本 | 异常点幅度 |
| `--seed` | `42` | 随机种子 |
| `--no_copy_original` | 关闭 | 不复制原始样本，仅输出合成样本 |

---

## 12. 每个代码文件的功能

| 文件 | 功能 |
|---|---|
| `README.md` | 项目说明、命令示例、接口和指标说明 |
| `雷达航迹分类接口.md` | 面向外部调用方的四接口说明文档 |
| `SERVICE_API.md` | 服务接口补充说明 |
| `3D_检测接口(1).md` | 3D 检测接口说明，非本雷达航迹分类主线 |
| `dataset_service.py` | 接口一；检查数据集是否为类别子目录加 `.mat` 文件 |
| `model_service.py` | 接口二；解析模型目录、读取权重/meta、构造 Transformer、校验模型和数据 |
| `attack_service.py` | 接口三；统一调用自然噪声、白盒攻击、黑盒攻击，保存扰动后的 `.mat`，写入核心评估 sidecar |
| `eval_service.py` | 接口四；输出分类鲁棒性指标，优先读取接口三生成的 `_interface_eval_result.json` |
| `model.py` | 定义 `RadarTrackTransformer` 和位置编码 |
| `data_utils.py` | 定义 `TrackSample`、12 通道字段、预处理、重采样和批处理转换 |
| `mat_loader.py` | 读取 MATLAB v7.3/HDF5 `.mat`，构造样本列表，支持训练/验证划分 |
| `train.py` | 训练 Transformer 分类模型，支持类别均衡、验证集、混淆矩阵、meta 保存 |
| `evaluate.py` | 普通测试集评估，输出 accuracy、macro 指标、classification mAP、混淆矩阵 |
| `metrics_utils.py` | 分类 mAP、百分比格式化、SSIM/PSNR 等信号质量指标 |
| `device_utils.py` | 统一选择 CUDA/CPU，打印设备信息 |
| `noise_perturbation.py` | 相对扰动投影、相对变化统计、三类基础自然噪声 |
| `robust_evaluate.py` | 基础自然噪声鲁棒性评估 |
| `robust_evaluate_enhanced.py` | 增强自然扰动评估，包含脉冲、漂移、通道衰减等 |
| `whitebox_attacks.py` | 白盒攻击核心实现和 CLI，包含 FGSM、PGD、CW |
| `blackbox_attacks.py` | 黑盒攻击核心实现和 CLI，包含 Square、NES、Transfer |
| `generate_synthetic_mat_data.py` | 基于已有 `.mat` 航迹生成合成训练样本 |
| `analyze_mat_to_table.py` | 将 `.mat` 航迹字段导出为 CSV 表格，便于人工检查 |
| `convert_mat_to_custom_dataset.py` | 将 MAT 航迹转换为自定义点数据格式的辅助脚本 |
| `requirements.txt` | Python 依赖列表 |
| `radarr-modify.code-workspace` | VS Code 工作区配置 |

---

## 13. 重要口径说明

### 分类指标不是 3D 检测指标

本项目是雷达航迹分类，主指标建议看：

- `accuracy`
- `macro_f1`
- `classification_mAP`
- `attack_success_rate`
- `robust_accuracy`

不要把这里的 `classification_mAP` 与 3D 检测任务中的 AP/BEV AP/3D AP 混为一谈。

### 接口三的 `max_rel_change`

接口三返回的 `max_rel_change` 是在模型输入的预处理张量空间统计的，和白盒/黑盒核心攻击代码的 `max_rel_change` 一致。

如果手动读取接口三保存后的 `.mat`，再按原始物理量字段计算相对变化，数值可能大于 `0.05`。这是因为模型输入前做过对数、归一化、z-score 等变换，原始物理空间和模型张量空间不是同一个尺度。

### Square Attack 的随机性

Square Attack 会随机选择时间窗口和扰动符号。不固定随机种子时，同样模型、同样数据、同样参数，多次运行的 `attack_success_rate` 可能略有波动。

### Python 交互窗口注意

在 Python REPL 中只输入代码，不要把提示符也复制进去：

```text
>>> 这是提示符，不要复制
... 这是多行提示符，也不要复制
```

如果修改了代码，已经打开的 Python 会话不会自动加载新代码。请先：

```python
exit()
```

再重新运行：

```powershell
python
```

---

## 14. 常用命令速查

训练：

```powershell
python train.py --mat_train_dir project_data/mat_train --save_path model_transformer.pth --epochs 100 --batch_size 64 --lr 0.001
```

评估：

```powershell
python evaluate.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/test --confusion
```

自然噪声：

```powershell
python robust_evaluate.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/test --condition all --max_rel_change 0.05 --worst_of_k 16
```

白盒 PGD：

```powershell
python whitebox_attacks.py --model_path dataoutput/model_transformer.pth --mat_test_dir project_data/test --attack pgd --max_rel_change 0.05
```

黑盒 Square：

```powershell
python blackbox_attacks.py --model_path dataoutput/model_aug_transformer.pth --mat_test_dir project_data/test --attack square --max_rel_change 0.05 --sq_max_queries 2000 --attack_restarts 1 --targeted_topk 1
```

接口调用：

```powershell
python -c "from attack_service import generate_adversarial; print(generate_adversarial('dataoutput/model_dir', 'project_data/test', 'project_data/test_adv_square', 'square'))"
python -c "from eval_service import evaluate_robustness; print(evaluate_robustness('dataoutput/model_dir', 'project_data/test', 'project_data/test_adv_square'))"
```
