# Watermark Remover

批量去水印工具，自动检测图片四角水印区域，使用 LaMa AI 模型修复。

## 安装

```bash
pip install -e .
```

## 使用

```bash
# 处理单张图片
watermark-remover image.jpg

# 批量处理目录
watermark-remover ./photos/

# 预览检测结果（红色标记水印区域）
watermark-remover ./photos/ --preview

# 调整检测参数
watermark-remover ./photos/ --corner-ratio 0.2 --threshold 20

# 不用 LaMa，用 OpenCV inpaint（快但效果差些）
watermark-remover ./photos/ --no-lama
```

## 原理

1. 扫描图片四角区域（默认 15%），通过边缘检测 + 高频分析定位水印
2. 生成精确的 mask 遮罩
3. 使用 LaMa 模型 AI 修复水印区域（首次运行自动下载模型）
4. 失败时回退到 OpenCV inpaint
