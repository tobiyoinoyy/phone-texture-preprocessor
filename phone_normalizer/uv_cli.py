"""CLI for the precision-cropped, consistently oriented six-view workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .api import NormalizerConfig
from .uv import process_six_views


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phone-normalizer-uv",
        description="BiRefNet_dynamic 六面精裁；可按已确认的数值基准对齐UV画布。",
    )
    parser.add_argument("input_dir", type=Path, help="含 front/back/left/right/top/bottom 六张图片的文件夹")
    parser.add_argument("--output-dir", type=Path, required=True, help="新结果目录，已有同名结果不会覆盖")
    parser.add_argument("--uv-profile", help="内置 iphone12pro-user-crop-v1 或自定义 JSON 路径；省略则只精裁")
    parser.add_argument("--max-anisotropy", type=float, default=0.05, help="UV 宽高缩放差异上限，默认0.05（试验值5%）；超限进入复核目录")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--birefnet-max-side", default="auto", help="推理最长边（默认1024）")
    parser.add_argument("--max-geometry-pixels", default="auto", help="几何阶段像素护栏（默认64MP）")
    parser.add_argument("--no-support-removal", action="store_true")
    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = process_six_views(
            args.input_dir, args.output_dir, uv_profile=args.uv_profile,
            max_anisotropy=args.max_anisotropy,
            config=NormalizerConfig(
                crop_threshold=128, device=args.device,
                birefnet_max_side=args.birefnet_max_side,
                max_geometry_pixels=args.max_geometry_pixels,
                remove_support=not args.no_support_removal,
                deskew=not args.no_deskew,
            ),
            progress=lambda face: print(f"处理中：{face}", flush=True),
        )
    except Exception as exc:
        print(f"处理失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for face, info in report["faces"].items():
        width, height = info["output_size"]
        label = "待复核" if face in report["review_required_faces"] else "完成"
        print(f"{label}：{info['output']}  {width}×{height}")
    print(f"结果：{args.output_dir.expanduser().resolve()}")
    if report["profile_id"]:
        print("对齐检查图位于 diagnostics/；几何检查通过仍需核对接口、按键与镜头的UV位置。")
        if report["review_required_faces"]:
            print("以下面仅输出复核预览，不能直接用于UV：" + "、".join(report["review_required_faces"]))
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
