"""图像修复模块 - LaMa 模型 + OpenCV 后备"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def inpaint_with_lama(image_path: Path, mask: np.ndarray, output_path: Path):
    """使用 IOPaint 的 LaMa 模型去水印"""
    mask_path = output_path.parent / f".mask_{output_path.stem}.png"
    cv2.imwrite(str(mask_path), mask)

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "iopaint", "run",
                "--model=lama",
                f"--image={image_path}",
                f"--mask={mask_path}",
                f"--output={output_path.parent}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"iopaint 运行失败: {result.stderr}")

        iopaint_output = output_path.parent / image_path.name
        if iopaint_output != output_path and iopaint_output.exists():
            iopaint_output.rename(output_path)
    finally:
        mask_path.unlink(missing_ok=True)


def inpaint_with_cv2(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """使用 OpenCV 内置 inpaint 作为后备方案"""
    return cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
