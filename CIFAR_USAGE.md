# CIFAR-10/100 Usage

The original `train.py` is left unchanged. Use `scripts/train_cifar.py` to run
CIFAR-10 or CIFAR-100 through the same training pipeline.

## Basic Commands

Train DiDiCM on CIFAR-10 with the default DiDiRN-18 backbone:

```bash
python scripts/train_cifar.py --dataset cifar10 --gpu 0 --amp
```

Train CIFAR-100 with a larger DiDiRN backbone:

```bash
python scripts/train_cifar.py --dataset cifar100 --model didirn34 --gpu 0 --amp
```

Use a specific CUDA device string instead of `CUDA_VISIBLE_DEVICES`:

```bash
python scripts/train_cifar.py --dataset cifar10 --device cuda:1 --amp
```

Run on CPU for a quick smoke test:

```bash
python scripts/train_cifar.py --dataset cifar10 --device cpu --epochs 1 --batch-size 8 --validation-batch-size 8
```

Preview the generated `train.py` command without running training:

```bash
python scripts/train_cifar.py --dataset cifar100 --model didirn50 --gpu 0 --dry-run
```

## Important Options

- `--dataset`: `cifar10` or `cifar100`.
- `--model`: defaults to `didirn18`; diffusion mode supports `didirn18`,
  `didirn34`, `didirn50`, `didirn101`, and `didirn152`.
- `--gpu`: exposes one physical GPU through `CUDA_VISIBLE_DEVICES`, then uses
  `--device cuda` inside `train.py`.
- `--device`: passed directly to `train.py`; useful for `cuda:0`, `cuda:1`, or
  `cpu`.
- `--log-dir`: defaults to `./logs/cifar`. Runtime logger output is written to
  `./logs/cifar/<experiment>/log.txt` while the terminal still shows one copy.
- `--no-log-file`: disables file logging from the CIFAR launcher.
- `--checkpoint-policy`: defaults to `best`, so only `model_best.pth.tar` is
  kept. Use `none` to disable checkpoint files, or `all` to keep epoch history.
- `--no-diffusion`: disables DiDiCM diffusion mode. Use this only when running a
  standard timm model such as `resnet18`.
- The wrapper uses conservative DiDiCM defaults for CIFAR: `--lr 5e-4`,
  `--warmup-epochs 10`, `--clip-grad 1.0`, and `--diffusion-eps 1e-2`.
  The original ImageNet-style `--lr 0.1`, and even `0.005` in early tests, can
  make the CL diffusion loss overflow on CIFAR.
- DiDiRN disables its original maxpool stem by default on CIFAR through
  `--model-kwargs no_maxpool=True`, preserving more spatial resolution for
  32x32 images. Add `--imagenet-stem` to keep the original stem.
- Keep `--workers` at `1` or higher. The timm DataLoader path used here does
  not accept `--workers 0`.
- Do not pass `--data-ratio` for CIFAR. The original subset helper only accepts
  timm `ImageDataset`, while CIFAR is loaded through torchvision.
- Any unknown extra arguments are forwarded to `train.py`, for example:

```bash
python scripts/train_cifar.py --dataset cifar10 --model didirn18 --gpu 0 --mixup 0.2 --cutmix 1.0
```

## What the Wrapper Sets

The wrapper maps CIFAR-specific defaults onto the existing training arguments:

- `--dataset torch/cifar10` or `--dataset torch/cifar100`
- `--num-classes 10` or `--num-classes 100`
- `--input-size 3 32 32`
- CIFAR-specific mean and standard deviation
- `--train-split train`
- `--val-split validation`
- `--dataset-download` unless `--no-download` is set
