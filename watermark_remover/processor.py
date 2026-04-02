"""图片处理模块 - 单张/批量处理逻辑"""

from pathlib import Path

import cv2
import numpy as np

from .detector import detect_watermark_mask
from .inpainter import inpaint_with_cv2, inpaint_with_lama

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def process_image(
    image_path: Path,
    output_path: Path,
    corner_ratio: float = 0.15,
    threshold: int = 30,
    padding: int = 10,
    preview: bool = False,
    use_lama: bool = True,
) -> bool:
    """处理单张图片，返回是否成功"""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [x] 无法读取: {image_path}")
        return False

    print(f"  [>] 检测水印: {image_path.name} ({image.shape[1]}x{image.shape[0]})")

    mask, found = detect_watermark_mask(image, corner_ratio, threshold, padding)

    if not found or np.sum(mask) == 0:
        print("    [!] 未检测到明显水印，将跳过此图片")
        return False

    watermark_ratio = np.sum(mask > 0) / mask.size * 100
    print(f"    水印区域: {watermark_ratio:.1f}% 的图片面积")

    if preview:
        preview_img = image.copy()
        overlay = preview_img.copy()
        overlay[mask > 0] = [0, 0, 255]
        preview_img = cv2.addWeighted(overlay, 0.4, preview_img, 0.6, 0)
        cv2.imwrite(str(output_path), preview_img)
        print(f"    预览已保存: {output_path}")
        return True

    if use_lama:
        try:
            print("    使用 LaMa 模型修复中...")
            inpaint_with_lama(image_path, mask, output_path)
            print(f"    [OK] 保存: {output_path}")
            return True
        except Exception as e:
            print(f"    [!] LaMa 失败 ({e})，回退到 OpenCV inpaint")

    print("    使用 OpenCV inpaint 修复中...")
    result = inpaint_with_cv2(image, mask)
    cv2.imwrite(str(output_path), result)
    print(f"    [OK] 保存: {output_path}")
    return True


def process_directory(
    input_dir: Path,
    output_dir: Path,
    corner_ratio: float = 0.15,
    threshold: int = 30,
    padding: int = 10,
    preview: bool = False,
    use_lama: bool = True,
) -> tuple[int, int]:
    """批量处理目录，返回 (成功数, 总数)"""
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
    )

    if not images:
        print(f"目录中没有找到支持的图片: {input_dir}")
        return 0, 0

    print(f"找到 {len(images)} 张图片，输出到: {output_dir}\n")

    success = 0
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}]")
        output_path = output_dir / f"clean_{img_path.name}"
        if process_image(
            img_path, output_path, corner_ratio, threshold, padding, preview, use_lama
        ):
            success += 1
        print()

    return success, len(images)
