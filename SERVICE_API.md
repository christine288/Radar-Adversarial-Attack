# 鲁棒性评估 Service 接口运行说明

本文档说明 `dataset_service.py`、`model_service.py`、`attack_service.py`、`eval_service.py` 四个接口的**调用方式**、**数据格式**及与仓库原有代码（`model.py`、`evaluate.py`、`whitebox_attacks.py` 等）的衔接关系。

> 所有示例均在**项目根目录** `E:\github\Radar-Adversarial-Attack` 下执行。PowerShell 请使用**单行**命令，避免在参数中间换行。

---

## 一、与原有代码的关系

| Service 文件 | 主要函数 | 实际调用的仓库模块 |
|--------------|----------|-------------------|
| `dataset_service.py` | `prepare_dataset` | 独立实现（目录结构校验 + 生成 `ImageSets/val.txt`） |
| `model_service.py` | `validate_model` | **`model.RadarTrackTransformer`** + `torch.load`；数据集走 `prepare_dataset` |
| `attack_service.py` | `generate_adversarial` | **MAT 航迹**：`mat_loader`、`whitebox_attacks`、`blackbox_attacks`、`noise_perturbation`、`model`；**3D 点云**：占位式高斯扰动（未接 3D 检测模型） |
| `eval_service.py` | `evaluate_robustness` | **`evaluate.evaluate_model`**（内部用 `model.py` + `mat_loader` + `metrics_utils`） |

**结论**：雷达 **MAT 分类** 链路已接到 `model.py` / `evaluate.py` / 白盒·黑盒攻击脚本；**KITTI / Custom 点云** 仅做目录校验与简单噪声扰动，**未**接入真实 3D 检测网络。接口文档里的字段名（如 `mAP_3D`、`chamfer`）在 MAT 场景下为**字段兼容映射**（分类 mAP 写入 `mAP_3D` 键）。

---

## 二、支持的数据格式

### 1. 雷达 MAT（本仓库主流程，推荐）

```
<dataset_path>/
  类别A/
    *.mat
  类别B/
    *.mat
```

- 不要求 `dataset.yaml`。
- `attack_service` / `eval_service` 会走 **MAT 专用分支**（12 通道航迹 + `RadarTrackTransformer`）。

### 2. Custom 点云（接口文档格式）

```
<dataset_path>/
  dataset.yaml
  points/*.npy      # N×4
  labels/
  ImageSets/val.txt # 缺失时自动生成
```

### 3. KITTI 点云（接口文档格式）

```
<dataset_path>/
  training/velodyne/*.bin
  training/calib/
  training/label_2/
  ImageSets/val.txt
```

---

## 三、环境

```powershell
cd E:\github\Radar-Adversarial-Attack
pip install -r requirements.txt
```

---

## 四、接口一：`prepare_dataset`

**文件**：`dataset_service.py`  
**函数**：`prepare_dataset(dataset_path: str) -> bool`

### Python 调用

```powershell
python -c "from dataset_service import prepare_dataset; ok=prepare_dataset('project_data/mat_test'); print('OK' if ok else 'FAIL')"
```

Custom / KITTI 示例：

```powershell
python -c "from dataset_service import prepare_dataset; print(prepare_dataset('path/to/kitti_or_custom_dataset'))"
```

### 行为说明

- 检查目录结构是否合法。
- 若缺少 `ImageSets/val.txt`，会根据 `points/` 或 `training/velodyne/` 下的文件名自动生成。
- **MAT 按类别子目录组织时**，本函数会返回 `False`（未识别为 KITTI/Custom）；MAT 数据集**不需要**调用此接口，可直接用于 `attack_service` / `eval_service`。

---

## 五、接口二：`validate_model`

**文件**：`model_service.py`  
**函数**：`validate_model(model_path: str, dataset_path: str) -> bool`

### 模型路径规则

- 传入 **`.pth` / `.pt` 文件**；或
- 传入 **目录**，且目录内**有且仅有 1 个** `.pth` 或 `.pt`。

### Python 调用（MAT 示例）

```powershell
python -c "from model_service import validate_model; ok=validate_model('dataoutput/model_aug_transformer.pth', 'project_data/mat_test_augmented'); print('OK' if ok else 'FAIL')"
```

### 校验内容

1. 权重能否 `torch.load`，且含 `input_size`、`model_state_dict`。
2. 能否实例化并加载 **`RadarTrackTransformer`**（`model.py`）。
3. `prepare_dataset(dataset_path)` 是否为 True（MAT 目录会在此步失败；MAT 场景可跳过本接口，直接用 `attack_service` 内置校验）。

---

## 六、接口三：`generate_adversarial`

**文件**：`attack_service.py`  
**函数**：`generate_adversarial(model_path, dataset_path, adv_dataset_path, attack_method) -> dict`

### 支持的 `attack_method`（MAT 数据集）

| 类型 | 取值 |
|------|------|
| 白盒 | `fgsm`, `pgd`, `cw`, `deepfool` |
| 黑盒 | `square`, `nes`, `transfer` |
| 自然噪声 | `gaussian`, `salt_pepper`, `speckle` |

点云（KITTI / Custom）仅根据方法名调整**扰动强度系数**，不调用 `model.py`。

### Python 调用（MAT + FGSM 示例）

```powershell
python -c "from attack_service import generate_adversarial; r=generate_adversarial('dataoutput/model_aug_transformer.pth', 'project_data/mat_test_augmented', 'project_data/mat_test_adv_fgsm', 'fgsm'); print(r)"
```

### Python 调用（MAT + PGD 示例）

```powershell
python -c "from attack_service import generate_adversarial; r=generate_adversarial('dataoutput/model_aug_transformer.pth', 'project_data/mat_test_augmented', 'project_data/mat_test_adv_pgd', 'pgd'); print(r)"
```

### 返回 `result_dict`（相似度类指标）

```python
{
    "mean_L2": ...,   # 平均扰动距离（MAT 为逆变换后航迹空间 L2）
    "max_L2": ...,
    "mad": ...,       # 平均失真度
    "chamfer": ...,   # 当前实现与 mean_L2 同类统计（占位）
    "pts_ratio": 1.0  # MAT 固定为 1.0（点数不变）
}
```

对抗样本写入 `adv_dataset_path`，**目录结构与源 MAT 数据集一致**（按类别子目录 + 同名 `.mat`）。

---

## 七、接口四：`evaluate_robustness`

**文件**：`eval_service.py`  
**函数**：`evaluate_robustness(model_path, dataset_path, adv_dataset_path) -> dict`

### Python 调用

```powershell
python -c "from eval_service import evaluate_robustness; r=evaluate_robustness('dataoutput/model_aug_transformer.pth', 'project_data/mat_test_augmented', 'project_data/mat_test_adv_fgsm'); print(r)"
```

### 返回 `result_dict`

```python
{
    "clean": {
        "mAP_3D": 0.4258   # 实际为分类 mAP（evaluate.py）
    },
    "Adversarial": {
        "mAP_3D": 0.4028,
        "mAP_drop": 0.0230,
        "mAP_retention_rate": 0.9456,
        "attack_success_rate": 0.0528   # 干净预测正确子集上的翻转率
    }
}
```

内部对**干净集**、**对抗集**各调用一次 `evaluate.evaluate_model`；`attack_success_rate` 与 `whitebox_attacks.py` / `blackbox_attacks.py` 中 ISR 含义一致。

---

## 八、完整 MAT 流水线示例

在 PowerShell 中依次执行（路径按本机修改）：

```powershell
cd E:\github\Radar-Adversarial-Attack

python -c "from model_service import validate_model; print('validate:', validate_model('dataoutput/model_aug_transformer.pth', 'project_data/mat_test_augmented'))"

python -c "from attack_service import generate_adversarial; print(generate_adversarial('dataoutput/model_aug_transformer.pth', 'project_data/mat_test_augmented', 'project_data/mat_test_adv_pgd', 'pgd'))"

python -c "from eval_service import evaluate_robustness; import json; print(json.dumps(evaluate_robustness('dataoutput/model_aug_transformer.pth', 'project_data/mat_test_augmented', 'project_data/mat_test_adv_pgd'), indent=2))"
```

也可保存为脚本 `run_mat_pipeline.py`：

```python
from attack_service import generate_adversarial
from eval_service import evaluate_robustness

MODEL = "dataoutput/model_aug_transformer.pth"
CLEAN = "project_data/mat_test_augmented"
ADV = "project_data/mat_test_adv_pgd"
METHOD = "pgd"

print("generate:", generate_adversarial(MODEL, CLEAN, ADV, METHOD))
print("evaluate:", evaluate_robustness(MODEL, CLEAN, ADV))
```

```powershell
python run_mat_pipeline.py
```

---

## 九、与 CLI 脚本的对应关系

若不需要平台四接口封装，仍可直接使用原有命令行：

| 需求 | 原脚本 |
|------|--------|
| 测试准确率 | `python evaluate.py --model_path ... --mat_test_dir ...` |
| 白盒攻击评估 | `python whitebox_attacks.py --attack pgd ...` |
| 黑盒攻击评估 | `python blackbox_attacks.py --attack square ...` |
| 自然噪声鲁棒性 | `python robust_evaluate.py --condition all ...` |

`attack_service` + `eval_service` 相当于把「生成对抗 MAT + 对比 evaluate」打包成平台要求的函数签名。

---

## 十、常见问题

1. **`ImportError: attempted relative import`**  
   请确保 `model_service.py` 使用 `from dataset_service import prepare_dataset`（不要用 `from .dataset_service`），并在项目根目录执行。

2. **`validate_model` 对 MAT 返回 False**  
   `validate_model` 末尾调用 `prepare_dataset`，MAT 目录不符合 KITTI/Custom 规范时会失败。**MAT 攻击/评估可跳过该步**，直接调用 `generate_adversarial` / `evaluate_robustness`（内部对 MAT 有单独分支）。

3. **权重路径**  
   使用 `dataoutput/xxx.pth`，且建议同目录存在 `xxx.pth.meta.json`（含 `mat_target_points` 等）。

4. **3D 点云数据集**  
   `prepare_dataset` 可用；`generate_adversarial` 做点云复制 + 高斯扰动；**不会**加载 3D 检测模型，指标与文档中的 3D mAP 含义不同。

5. **设备**  
   Service 层 MAT 攻击/评估当前默认 **`device="cpu"`**；大批量可在后续改为 `cuda`（需改 `attack_service.py` / `eval_service.py` 内传参）。
