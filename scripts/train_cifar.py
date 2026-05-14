#!/usr/bin/env python3
"""CIFAR-10/100 launcher for the original DiDiCM training script.

This wrapper keeps train.py unchanged and maps CIFAR-specific defaults onto
the existing timm-based arguments. Extra arguments are forwarded to train.py.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CIFAR_META = {
    "cifar10": {
        "dataset": "torch/cifar10",
        "num_classes": 10,
        "mean": ("0.4914", "0.4822", "0.4465"),
        "std": ("0.2470", "0.2435", "0.2616"),
    },
    "cifar100": {
        "dataset": "torch/cifar100",
        "num_classes": 100,
        "mean": ("0.5071", "0.4867", "0.4408"),
        "std": ("0.2675", "0.2565", "0.2761"),
    },
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch train.py on CIFAR-10/100 without modifying the original training code.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=sorted(CIFAR_META), default="cifar10")
    parser.add_argument("--data-dir", default="./data", help="Root folder for torchvision CIFAR data.")
    parser.add_argument("--no-download", action="store_true", help="Do not let torchvision download CIFAR.")
    parser.add_argument(
        "--model",
        default="didirn18",
        help=(
            "Backbone/model name. Diffusion mode requires a DiDiRN model "
            "(didirn18/34/50/101/152). Use --no-diffusion for standard timm models."
        ),
    )
    parser.add_argument("--img-size", type=int, default=32, help="Square CIFAR input resolution.")
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    parser.add_argument("--validation-batch-size", "-vb", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--workers", "-j", type=int, default=4)
    parser.add_argument("--opt", default="sgd")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--sched", default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./output/cifar")
    parser.add_argument("--experiment", default=None)
    parser.add_argument(
        "--checkpoint-policy",
        choices=("all", "best", "none"),
        default="best",
        help="Checkpoint saving policy passed to train.py.",
    )
    parser.add_argument("--log-dir", default="./logs/cifar", help="Root folder for train.py realtime log files.")
    parser.add_argument("--log-file", default="log.txt", help="Log filename inside the experiment log folder.")
    parser.add_argument("--no-log-file", action="store_true", help="Do not add train.py file logging arguments.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to train.py, for example cuda, cuda:0, cuda:1, or cpu.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Expose only this physical CUDA GPU via CUDA_VISIBLE_DEVICES and pass --device cuda.",
    )
    parser.add_argument("--amp", action="store_true", help="Enable native PyTorch AMP.")
    parser.add_argument(
        "--imagenet-stem",
        action="store_true",
        help="Keep DiDiRN's original maxpool stem. By default CIFAR disables it for 32x32 inputs.",
    )
    parser.add_argument("--no-diffusion", action="store_true", help="Disable DiDiCM diffusion training.")
    parser.add_argument("--diffusion-sampler", choices=("cl", "cp"), default="cl")
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--diffusion-eps", type=float, default=1e-2)
    parser.add_argument("--dry-run", action="store_true", help="Print the generated command without running it.")
    return parser.parse_known_args()


def build_command(args: argparse.Namespace, extra_args: list[str]) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    train_py = repo_root / "train.py"
    meta = CIFAR_META[args.dataset]

    if args.workers < 1:
        raise SystemExit("CIFAR training uses timm's DataLoader path, which requires --workers >= 1.")
    if any(arg == "--data-ratio" or arg.startswith("--data-ratio=") for arg in extra_args):
        raise SystemExit(
            "--data-ratio is not supported for torchvision CIFAR datasets by the original train.py. "
            "Use the full CIFAR split or create a separate subset dataset wrapper."
        )
    if not args.no_diffusion and not args.model.lower().startswith("didirn"):
        raise SystemExit(
            "Diffusion training calls model(x, c=labels, t=sigma), so --model must be one of "
            "didirn18/didirn34/didirn50/didirn101/didirn152. "
            "For vanilla timm models, add --no-diffusion."
        )

    command = [
        sys.executable,
        str(train_py),
        "--dataset",
        meta["dataset"],
        "--data-dir",
        args.data_dir,
        "--train-split",
        "train",
        "--val-split",
        "validation",
        "--num-classes",
        str(meta["num_classes"]),
        "--model",
        args.model,
        "--input-size",
        "3",
        str(args.img_size),
        str(args.img_size),
        "--img-size",
        str(args.img_size),
        "--crop-pct",
        "1.0",
        "--mean",
        *meta["mean"],
        "--std",
        *meta["std"],
        "--batch-size",
        str(args.batch_size),
        "--validation-batch-size",
        str(args.validation_batch_size),
        "--epochs",
        str(args.epochs),
        "--workers",
        str(args.workers),
        "--opt",
        args.opt,
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--clip-grad",
        str(args.clip_grad),
        "--sched",
        args.sched,
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--seed",
        str(args.seed),
        "--output",
        args.output,
        "--checkpoint-policy",
        args.checkpoint_policy,
        "--device",
        "cuda" if args.gpu is not None else args.device,
    ]

    if not args.no_download:
        command.append("--dataset-download")
    if args.amp:
        command.append("--amp")
    if args.experiment:
        command.extend(["--experiment", args.experiment])
    if not args.no_log_file:
        command.extend(["--log-dir", args.log_dir, "--log-file", args.log_file])
    if not args.no_diffusion and args.model.lower().startswith("didirn") and not args.imagenet_stem:
        command.extend(["--model-kwargs", "no_maxpool=True"])
    if not args.no_diffusion:
        command.extend(
            [
                "--diffusion-enabled",
                "--diffusion-sampler",
                args.diffusion_sampler,
                "--diffusion-steps",
                str(args.diffusion_steps),
                "--diffusion-eps",
                str(args.diffusion_eps),
            ]
        )

    command.extend(extra_args)
    return command


def main() -> int:
    args, extra_args = parse_args()
    command = build_command(args, extra_args)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("Running command:")
    print(" ".join(command))
    if args.gpu is not None:
        print(f"CUDA_VISIBLE_DEVICES={args.gpu}")

    if args.dry_run:
        return 0
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
