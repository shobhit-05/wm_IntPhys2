#!/usr/bin/env python3
"""Minimal IntPhys2 subset inference for VideoMAEv2-Base checkpoints."""

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make imports robust when launched as `python scripts/xxx.py` via Slurm/srun.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.loader import get_video_loader
from models.modeling_finetune import (
    vit_base_patch16_224,
    vit_giant_patch14_224,
    vit_huge_patch16_224,
    vit_large_patch16_224,
    vit_small_patch16_224,
)

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:  # pragma: no cover
    load_safetensors = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "IntPhys2 subset inference with VideoMAEv2-Base"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Text file with one absolute video path per line.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for per-video outputs and run metadata.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Checkpoint path (.safetensors, .pt, or .pth).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vit_base_patch16_224",
        help="timm model name for this checkpoint.",
    )
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--max_videos", type=int, default=64)
    parser.add_argument(
        "--per_video_subdir",
        type=str,
        default="",
        help="Optional subdirectory under output_dir for per-video .pt files (e.g., per_video_embeddings).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_manifest(manifest_path: Path, max_videos: int) -> list[Path]:
    paths: list[Path] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paths.append(Path(line))
            if len(paths) >= max_videos:
                break
    return paths


def build_repo_model(model_name: str, input_size: int, clip_len: int) -> torch.nn.Module:
    builders = {
        "vit_small_patch16_224": vit_small_patch16_224,
        "vit_base_patch16_224": vit_base_patch16_224,
        "vit_large_patch16_224": vit_large_patch16_224,
        "vit_huge_patch16_224": vit_huge_patch16_224,
        "vit_giant_patch14_224": vit_giant_patch14_224,
    }
    if model_name not in builders:
        raise ValueError(
            f"Unsupported model_name={model_name}. Supported: {sorted(builders.keys())}"
        )
    # NOTE: this repo's VisionTransformer __init__ unconditionally touches
    # self.head.weight/self.head.bias during init. With num_classes=0 the head
    # becomes nn.Identity and construction crashes. Use a dummy classifier head;
    # embeddings are extracted via forward_features so head is not used.
    return builders[model_name](
        img_size=input_size,
        num_classes=1,
        all_frames=clip_len,
        tubelet_size=2,
        drop_path_rate=0.0,
        use_mean_pooling=True,
    )


def sample_uniform_indices(num_frames: int, clip_len: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("Video has zero frames.")
    if clip_len <= 0:
        raise ValueError("clip_len must be positive.")
    if num_frames == 1:
        return np.zeros((clip_len,), dtype=np.int64)
    return np.linspace(0, num_frames - 1, num=clip_len, dtype=np.int64)


def resize_shorter_side(frames_tchw: torch.Tensor, short_side: int) -> torch.Tensor:
    t, c, h, w = frames_tchw.shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid frame shape: {(t, c, h, w)}")
    if h < w:
        new_h = short_side
        new_w = int(round(w * (short_side / h)))
    else:
        new_w = short_side
        new_h = int(round(h * (short_side / w)))
    return F.interpolate(
        frames_tchw, size=(new_h, new_w), mode="bilinear", align_corners=False
    )


def center_crop(frames_tchw: torch.Tensor, crop_size: int) -> torch.Tensor:
    _, _, h, w = frames_tchw.shape
    if h < crop_size or w < crop_size:
        raise ValueError(
            f"Cannot center-crop {crop_size} from resized shape {(h, w)}."
        )
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return frames_tchw[:, :, top : top + crop_size, left : left + crop_size]


def preprocess_frames(frames_thwc_uint8: torch.Tensor, input_size: int) -> torch.Tensor:
    # THWC uint8 -> TCHW float in [0, 1]
    x = frames_thwc_uint8.permute(0, 3, 1, 2).contiguous().to(torch.float32) / 255.0
    x = resize_shorter_side(x, short_side=input_size)
    x = center_crop(x, crop_size=input_size)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    x = (x - mean) / std
    # TCHW -> CTHW for VideoMAE.
    return x.permute(1, 0, 2, 3).contiguous()


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> dict:
    def looks_like_safetensors(path: Path) -> bool:
        if path.suffix.lower() == ".safetensors":
            return True
        try:
            with path.open("rb") as f:
                header_len_raw = f.read(8)
                if len(header_len_raw) != 8:
                    return False
                header_len = int.from_bytes(header_len_raw, "little")
                if header_len <= 0 or header_len > 100_000_000:
                    return False
                first_header_char = f.read(1)
                return first_header_char == b"{"
        except Exception:
            return False

    if looks_like_safetensors(ckpt_path):
        if load_safetensors is None:
            raise RuntimeError(
                "safetensors is not available but a .safetensors checkpoint was provided."
            )
        state = load_safetensors(str(ckpt_path))
    else:
        # PyTorch >=2.6 changed torch.load default behavior (weights_only=True).
        try:
            state = torch.load(str(ckpt_path), map_location="cpu")
        except Exception:
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model", "module", "state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
    cleaned = {}
    for k, v in state.items():
        if k.startswith("model."):
            cleaned[k[len("model.") :]] = v
        elif k.startswith("module."):
            cleaned[k[len("module.") :]] = v
        else:
            cleaned[k] = v
    strict_used = True
    try:
        msg = model.load_state_dict(cleaned, strict=True)
    except RuntimeError as err:
        strict_used = False
        msg = model.load_state_dict(cleaned, strict=False)
        print(f"[warn] strict checkpoint load failed; retried with strict=False: {err}")
    info = {
        "strict_used": strict_used,
        "missing_keys": list(msg.missing_keys),
        "unexpected_keys": list(msg.unexpected_keys),
    }
    print(
        "Checkpoint load summary:",
        f"strict_used={info['strict_used']}",
        f"missing={len(info['missing_keys'])}",
        f"unexpected={len(info['unexpected_keys'])}",
    )
    return info


def extract_patch_mean_embedding(
    model: torch.nn.Module, input_bcthw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        b = input_bcthw.size(0)
        x = model.patch_embed(input_bcthw)
        if model.pos_embed is not None:
            x = x + model.pos_embed.expand(b, -1, -1).type_as(x).to(x.device).clone().detach()
        x = model.pos_drop(x)
        for blk in model.blocks:
            x = blk(x)
        patch_token_mean = x.mean(1)
        if model.fc_norm is not None:
            embedding = model.fc_norm(patch_token_mean)
        else:
            embedding = patch_token_mean
    return patch_token_mean, embedding


def collect_run_env() -> dict:
    env = {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_name_0"] = torch.cuda.get_device_name(0)
    return env


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    # Keep absolute path without dereferencing symlinks; HF snapshots use symlinks
    # that may drop the original .safetensors suffix after resolve().
    ckpt_path = Path(os.path.abspath(str(Path(args.ckpt_path).expanduser())))
    output_dir.mkdir(parents=True, exist_ok=True)
    per_video_dir = output_dir / args.per_video_subdir if args.per_video_subdir else output_dir
    per_video_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_paths = load_manifest(manifest_path, max_videos=args.max_videos)
    if not video_paths:
        raise RuntimeError(f"No videos found in manifest: {manifest_path}")
    missing = [str(p) for p in video_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} video path(s) from manifest missing. First: {missing[0]}")

    model = build_repo_model(
        model_name=args.model_name,
        input_size=args.input_size,
        clip_len=args.clip_len,
    )
    ckpt_info = load_checkpoint(model, ckpt_path)
    model.eval().to(device)
    video_loader = get_video_loader()

    run_meta = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "checkpoint": str(ckpt_path),
        "model_name": args.model_name,
        "clip_len": args.clip_len,
        "input_size": args.input_size,
        "sampling_rule": f"uniform linspace over [0, num_frames-1] with clip_len={args.clip_len}",
        "decode_backend": "decord.VideoReader via dataset.loader.get_video_loader",
        "preprocess": {
            "scale": "uint8_to_float32_div_255",
            "resize": "shorter_side_to_224_bilinear",
            "crop": "center_crop_224x224",
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "layout_model_input": "B,C,T,H,W",
        },
        "checkpoint_load_info": ckpt_info,
        "run_env": collect_run_env(),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    summary_path = output_dir / "summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as sf:
        for idx, video_path in enumerate(video_paths):
            try:
                vr = video_loader(str(video_path))
                total_frames = len(vr)
                indices = sample_uniform_indices(total_frames, args.clip_len)
                raw_np = vr.get_batch(indices).asnumpy()  # [T, H, W, C], uint8
                raw = torch.from_numpy(raw_np)
                clip_cthw = preprocess_frames(raw, input_size=args.input_size)
                model_input = clip_cthw.unsqueeze(0).to(device)  # [1, C, T, H, W]
                patch_mean, embedding = extract_patch_mean_embedding(model, model_input)
            except Exception as err:
                raise RuntimeError(f"Failed on video: {video_path}") from err

            video_id = video_path.stem
            out_path = per_video_dir / f"{video_id}.pt"
            payload = {
                "video_id": video_id,
                "video_path": str(video_path),
                "total_frames": int(total_frames),
                "sampled_frame_indices": indices.tolist(),
                "embedding": embedding.detach().cpu(),  # [1, D]
                "patch_token_mean_before_fc_norm": patch_mean.detach().cpu(),  # [1, D]
                "loss": None,
                "loss_note": "No reconstruction/inference loss in encoder-only forward path.",
                "shapes": {
                    "raw_frames_thwc": list(raw.shape),
                    "clip_cthw": list(clip_cthw.shape),
                    "model_input_bcthw": list(model_input.shape),
                    "patch_token_mean": list(patch_mean.shape),
                    "embedding": list(embedding.shape),
                },
                "metadata": {
                    "clip_len": args.clip_len,
                    "input_size": args.input_size,
                    "sampling_rule": run_meta["sampling_rule"],
                    "decode_backend": run_meta["decode_backend"],
                    "model_name": args.model_name,
                    "checkpoint": str(ckpt_path),
                },
            }
            torch.save(payload, out_path)

            line = {
                "video_id": video_id,
                "video_path": str(video_path),
                "output_file": str(out_path),
                "sampled_frame_indices": indices.tolist(),
                "shapes": payload["shapes"],
                "loss": None,
            }
            sf.write(json.dumps(line) + "\n")

            print(
                f"[{idx + 1}/{len(video_paths)}] {video_id} "
                f"raw={payload['shapes']['raw_frames_thwc']} "
                f"input={payload['shapes']['model_input_bcthw']} "
                f"emb={payload['shapes']['embedding']} -> {out_path}"
            )

    print(f"Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
