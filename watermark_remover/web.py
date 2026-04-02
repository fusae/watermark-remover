"""Web UI - 基于 Gradio 的可视化去水印界面"""

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from .detector import detect_watermark_mask
from .inpainter import inpaint_with_cv2, inpaint_with_lama


def preview_mask(image: np.ndarray, corner_ratio: float, threshold: int, padding: int):
    """预览水印检测结果，红色标记水印区域"""
    if image is None:
        return None

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask, found = detect_watermark_mask(image_bgr, corner_ratio, threshold, padding)

    if not found or np.sum(mask) == 0:
        return image  # 没检测到水印，返回原图

    overlay = image.copy()
    overlay[mask > 0] = [255, 60, 60]
    preview = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    return preview


def remove_watermark(
    image: np.ndarray,
    corner_ratio: float,
    threshold: int,
    padding: int,
    use_lama: bool,
):
    """去水印并返回结果"""
    if image is None:
        return None

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask, found = detect_watermark_mask(image_bgr, corner_ratio, threshold, padding)

    if not found or np.sum(mask) == 0:
        gr.Info("未检测到水印")
        return image

    ratio = np.sum(mask > 0) / mask.size * 100
    gr.Info(f"检测到水印区域: {ratio:.1f}% 面积")

    if use_lama:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                input_path = tmp / "input.png"
                output_path = tmp / "output.png"
                cv2.imwrite(str(input_path), image_bgr)
                inpaint_with_lama(input_path, mask, output_path)
                result = cv2.imread(str(output_path))
                return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        except Exception as e:
            gr.Warning(f"LaMa 失败 ({e})，回退到 OpenCV")

    result = inpaint_with_cv2(image_bgr, mask)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def create_app() -> gr.Blocks:
    with gr.Blocks(title="去水印工具", theme=gr.themes.Soft()) as app:
        gr.Markdown("## 去水印工具\n拖拽上传图片，自动检测四角水印并 AI 修复")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="上传图片", type="numpy")
                with gr.Accordion("参数调整", open=False):
                    corner_ratio = gr.Slider(0.05, 0.35, value=0.15, step=0.01, label="角落区域比例")
                    threshold = gr.Slider(10, 80, value=30, step=1, label="检测灵敏度（越低越敏感）")
                    padding = gr.Slider(3, 30, value=10, step=1, label="Mask 膨胀像素")
                    use_lama = gr.Checkbox(value=True, label="使用 LaMa 模型（效果好但慢）")
                with gr.Row():
                    preview_btn = gr.Button("预览检测", variant="secondary")
                    remove_btn = gr.Button("去水印", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="结果", type="numpy")

        preview_btn.click(
            fn=preview_mask,
            inputs=[input_image, corner_ratio, threshold, padding],
            outputs=output_image,
        )
        remove_btn.click(
            fn=remove_watermark,
            inputs=[input_image, corner_ratio, threshold, padding, use_lama],
            outputs=output_image,
        )

    return app


def main():
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
