"""水印检测模块 - 自动检测图片四角的水印区域"""

import cv2
import numpy as np


def detect_watermark_mask(
    image: np.ndarray,
    corner_ratio: float = 0.15,
    threshold: int = 30,
    padding: int = 10,
) -> np.ndarray:
    """
    检测图片四角的水印区域，返回二值 mask。
    白色 (255) = 水印区域，黑色 (0) = 保留区域。
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h, w), dtype=np.uint8)

    ch = int(h * corner_ratio)
    cw = int(w * corner_ratio)
    corners = {
        "top_left": (0, 0, cw, ch),
        "top_right": (w - cw, 0, w, ch),
        "bottom_left": (0, h - ch, cw, h),
        "bottom_right": (w - cw, h - ch, w, h),
    }

    found_any = False
    for name, (x1, y1, x2, y2) in corners.items():
        roi = gray[y1:y2, x1:x2]
        roi_color = image[y1:y2, x1:x2]

        # 边缘检测
        edges = cv2.Canny(roi, 50, 150)

        # 高频细节检测
        laplacian = cv2.Laplacian(roi, cv2.CV_64F)
        lap_abs = np.uint8(np.absolute(laplacian))
        if len(lap_abs.shape) == 3:
            lap_abs = cv2.cvtColor(lap_abs, cv2.COLOR_BGR2GRAY)

        # 综合判断
        edge_density = np.sum(edges > 0) / edges.size
        lap_density = np.sum(lap_abs > threshold) / lap_abs.size
        has_watermark = edge_density > 0.02 or lap_density > 0.05

        if has_watermark:
            found_any = True
            combined = cv2.bitwise_or(edges, lap_abs)
            _, binary = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (padding * 2, padding * 2)
            )
            dilated = cv2.dilate(binary, kernel, iterations=2)

            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            corner_mask = np.zeros_like(roi, dtype=np.uint8)
            if len(corner_mask.shape) == 3:
                corner_mask = corner_mask[:, :, 0]

            min_area = roi.size * 0.001
            for cnt in contours:
                if cv2.contourArea(cnt) > min_area:
                    cv2.drawContours(corner_mask, [cnt], -1, 255, -1)

            corner_mask = cv2.dilate(
                corner_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding, padding)),
                iterations=1,
            )

            mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], corner_mask)

    return mask, found_any
