"""Web UI - 基于 Gradio 的可视化去水印界面"""

import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from watermark_remover.detector import detect_watermark_mask
    from watermark_remover.inpainter import inpaint_with_cv2, inpaint_with_lama
else:
    from .detector import detect_watermark_mask
    from .inpainter import inpaint_with_cv2, inpaint_with_lama


def preview_mask(
    image: np.ndarray,
    corner_ratio: float,
    threshold: int,
    padding: int,
    progress=gr.Progress(),
):
    """预览水印检测结果，红色标记水印区域"""
    if image is None:
        yield None, "请先上传图片"
        return

    progress(0.1, desc="正在读取图片")
    yield image, "正在读取图片"
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    progress(0.5, desc="正在检测水印区域")
    yield image, "正在检测水印区域"
    mask, found = detect_watermark_mask(image_bgr, corner_ratio, threshold, padding)

    if not found or np.sum(mask) == 0:
        progress(1.0, desc="未检测到水印")
        yield image, "未检测到水印"
        return

    progress(0.8, desc="正在生成预览")
    yield image, "正在生成预览"
    overlay = image.copy()
    overlay[mask > 0] = [255, 60, 60]
    preview = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    progress(1.0, desc="预览完成")
    yield preview, "预览完成"


def remove_watermark(
    image: np.ndarray,
    corner_ratio: float,
    threshold: int,
    padding: int,
    use_lama: bool,
    progress=gr.Progress(),
):
    """去水印并返回结果"""
    if image is None:
        yield None, "请先上传图片"
        return

    progress(0.05, desc="正在读取图片")
    yield image, "正在读取图片"
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    progress(0.25, desc="正在检测水印区域")
    yield image, "正在检测水印区域"
    mask, found = detect_watermark_mask(image_bgr, corner_ratio, threshold, padding)

    if not found or np.sum(mask) == 0:
        progress(1.0, desc="未检测到水印")
        gr.Info("未检测到水印")
        yield image, "未检测到水印"
        return

    ratio = np.sum(mask > 0) / mask.size * 100
    gr.Info(f"检测到水印区域: {ratio:.1f}% 面积")

    if use_lama:
        try:
            progress(0.45, desc="正在准备 LaMa 修复")
            yield image, "正在准备 LaMa 修复"
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                input_path = tmp / "input.png"
                output_path = tmp / "output.png"
                cv2.imwrite(str(input_path), image_bgr)
                progress(0.6, desc="LaMa 修复中")
                yield image, "LaMa 修复中"
                inpaint_with_lama(input_path, mask, output_path)
                result = cv2.imread(str(output_path))
                progress(1.0, desc="去水印完成")
                yield cv2.cvtColor(result, cv2.COLOR_BGR2RGB), "去水印完成"
                return
        except Exception as e:
            gr.Warning(f"LaMa 失败 ({e})，回退到 OpenCV")

    progress(0.75, desc="正在使用 OpenCV 修复")
    yield image, "正在使用 OpenCV 修复"
    result = inpaint_with_cv2(image_bgr, mask)
    progress(1.0, desc="去水印完成")
    yield cv2.cvtColor(result, cv2.COLOR_BGR2RGB), "去水印完成"


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="去水印工具",
        theme=gr.themes.Soft(),
        analytics_enabled=False,
    ) as app:
        gr.Markdown("## 去水印工具\n拖拽上传图片，自动检测四角水印并 AI 修复")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    type="numpy",
                    sources=["upload"],
                    show_label=False,
                )
                with gr.Accordion("参数调整", open=False):
                    corner_ratio = gr.Slider(0.05, 0.35, value=0.15, step=0.01, label="角落区域比例")
                    threshold = gr.Slider(10, 80, value=30, step=1, label="检测灵敏度（越低越敏感）")
                    padding = gr.Slider(3, 30, value=10, step=1, label="Mask 膨胀像素")
                    use_lama = gr.Checkbox(value=True, label="使用 LaMa 模型（效果好但慢）")
                with gr.Row():
                    preview_btn = gr.Button("预览检测", variant="secondary")
                    remove_btn = gr.Button("去水印", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="numpy", show_label=False)
                status_text = gr.Markdown("等待操作")

        preview_btn.click(
            fn=preview_mask,
            inputs=[input_image, corner_ratio, threshold, padding],
            outputs=[output_image, status_text],
            queue=True,
            show_progress="full",
            show_api=False,
        )
        remove_btn.click(
            fn=remove_watermark,
            inputs=[input_image, corner_ratio, threshold, padding, use_lama],
            outputs=[output_image, status_text],
            queue=True,
            show_progress="full",
            show_api=False,
        )

    return app


def main():
    app = create_app()
    app.queue(default_concurrency_limit=1)
    app.launch(
        show_api=False,
        server_name="127.0.0.1",
        server_port=7860,
    )


if __name__ == "__main__":
    main()
