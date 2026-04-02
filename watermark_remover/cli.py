"""命令行入口"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from watermark_remover.processor import (
        SUPPORTED_EXTENSIONS,
        process_directory,
        process_image,
    )
else:
    from .processor import SUPPORTED_EXTENSIONS, process_directory, process_image


def main():
    parser = argparse.ArgumentParser(
        description="批量去水印工具 - 自动检测四角水印并 AI 修复"
    )
    parser.add_argument("input", help="输入图片或目录路径")
    parser.add_argument("output", nargs="?", help="输出目录（默认: <input>/no_watermark）")
    parser.add_argument(
        "--corner-ratio", type=float, default=0.15,
        help="角落区域比例（默认: 0.15）",
    )
    parser.add_argument(
        "--threshold", type=int, default=30,
        help="检测灵敏度，越低越敏感（默认: 30）",
    )
    parser.add_argument(
        "--padding", type=int, default=10,
        help="mask 膨胀像素（默认: 10）",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="只生成 mask 预览，不执行去水印",
    )
    parser.add_argument(
        "--no-lama", action="store_true",
        help="不使用 LaMa 模型，直接用 OpenCV inpaint",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    use_lama = not args.no_lama

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"不支持的格式: {input_path.suffix}")
            sys.exit(1)

        if args.output:
            output_path = Path(args.output).resolve()
            if output_path.is_dir():
                output_path = output_path / f"clean_{input_path.name}"
        else:
            output_path = input_path.parent / f"clean_{input_path.name}"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        process_image(
            input_path, output_path,
            args.corner_ratio, args.threshold, args.padding,
            args.preview, use_lama,
        )

    elif input_path.is_dir():
        output_dir = Path(args.output).resolve() if args.output else input_path / "no_watermark"
        success, total = process_directory(
            input_path, output_dir,
            args.corner_ratio, args.threshold, args.padding,
            args.preview, use_lama,
        )
        if total > 0:
            print(f"完成！成功处理 {success}/{total} 张图片")
            print(f"输出目录: {output_dir}")

    else:
        print(f"路径不存在: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
