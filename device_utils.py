"""
统一设备选择：鲁棒性评估、预处理对比、训练等脚本默认优先使用 GPU（CUDA）。

用法：
  - 默认 `--device cuda`：若 `torch.cuda.is_available()` 为真则使用 GPU，否则回退 CPU 并给出警告。
  - 显式 `--device cpu` 强制 CPU。
  - `--require_gpu`：必须为 CUDA，否则直接报错（用于脚本级参数）。
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch


def resolve_device_str(
    user: Optional[str] = None,
    *,
    require_gpu: bool = False,
) -> str:
    """
    返回 PyTorch 可用的设备字符串，如 ``\"cuda\"``、``\"cpu\"``、``\"cuda:0\"``。

    Parameters
    ----------
    user
        ``None`` / ``\"cuda\"`` / ``\"gpu\"`` / ``\"auto\"``：优先 CUDA。
        ``\"cpu\"``：强制 CPU。
        ``\"cuda:0\"`` 等：在可用时使用该设备。
    require_gpu
        若为 True 且 CUDA 不可用，抛出 ``RuntimeError``。
    """
    u = (user or "cuda").strip().lower()
    if u in ("cuda", "gpu", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if require_gpu:
            raise RuntimeError(
                "已设置 --require_gpu，但当前环境未检测到 CUDA。"
                "请安装带 CUDA 的 PyTorch、NVIDIA 驱动，或使用 CPU：--device cpu"
            )
        warnings.warn(
            "未检测到 CUDA（torch.cuda.is_available() 为 False），已回退到 CPU。"
            "若需 GPU，请安装 CUDA 版 PyTorch 并正确安装显卡驱动。",
            UserWarning,
            stacklevel=2,
        )
        return "cpu"

    if u == "cpu":
        return "cpu"

    if u.startswith("cuda"):
        if not torch.cuda.is_available():
            if require_gpu:
                raise RuntimeError(f"请求设备 {u}，但 CUDA 不可用。")
            warnings.warn(f"请求 {u} 但 CUDA 不可用，已回退到 CPU。", UserWarning, stacklevel=2)
            return "cpu"
        return u

    # 其他字符串原样尝试（如未来 mps）
    return u


def print_device_banner(device_str: str) -> None:
    """在脚本开始时打印当前计算设备。"""
    print(f"[device] 使用计算设备: {device_str}")
    if device_str.startswith("cuda") and torch.cuda.is_available():
        try:
            print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
