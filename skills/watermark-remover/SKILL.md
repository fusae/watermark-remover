---
name: watermark-remover
description: Use this skill when the user wants to remove corner watermarks from local images or image folders with this repository's CLI. It supports single-image and batch directory processing, preview mask generation, and parameter tuning for corner ratio, threshold, padding, and LaMa fallback behavior.
---

# Watermark Remover

## Overview

This skill uses the local `watermark_remover` CLI in this repository to detect and remove watermarks from the four corners of images.

Use it for local files and folders when the user wants actual processing, preview masks, or batch cleanup. Do not use the web UI from this skill.

## When To Use

- The user wants to remove watermarks from one or more local images.
- The user provides a directory and wants batch processing.
- The user wants to preview detected watermark regions before running cleanup.
- The user needs parameter tuning for false positives or missed detections.

## Workflow

1. Confirm the input is a local image file or a directory of images.
2. If the request is ambiguous or the images may not have corner watermarks, run preview mode first.
3. Use `scripts/run_watermark_remover.py` for deterministic invocation from the repository root.
4. Report the output path and whether the run used preview, LaMa, or OpenCV fallback mode.

## Commands

Single image:

```bash
python3 skills/watermark-remover/scripts/run_watermark_remover.py /abs/path/image.jpg
```

Batch directory:

```bash
python3 skills/watermark-remover/scripts/run_watermark_remover.py /abs/path/photos /abs/path/output-dir
```

Preview only:

```bash
python3 skills/watermark-remover/scripts/run_watermark_remover.py /abs/path/photos --preview
```

Disable LaMa:

```bash
python3 skills/watermark-remover/scripts/run_watermark_remover.py /abs/path/photos --no-lama
```

Tune detection:

```bash
python3 skills/watermark-remover/scripts/run_watermark_remover.py /abs/path/photos --corner-ratio 0.2 --threshold 20 --padding 12
```

## Parameter Guidance

- `--corner-ratio`: Increase when the watermark sits farther from the corner or is physically larger.
- `--threshold`: Lower it when the detector misses faint watermarks; raise it when it over-detects textures.
- `--padding`: Increase it when cleanup leaves watermark edges behind.
- `--preview`: Preferred first step for uncertain inputs or large batches.
- `--no-lama`: Faster, but lower quality than the default LaMa path.

## Constraints

- This project is designed for corner watermarks, not center overlays or full-image marks.
- Inputs must be local filesystem paths.
- Supported formats come from the project CLI and processor module, not from the skill itself.
- If the environment is missing dependencies, install them from the repository root with `pip install -e .`.

## Resource

### scripts/run_watermark_remover.py

Use this wrapper instead of rebuilding CLI commands manually. It runs the local module from the repository root and exposes the same tuning options as the project CLI.
