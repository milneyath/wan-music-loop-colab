"""Wan2.2 image-to-video music-loop engine.

Pure, importable functions — no Colab-specific code lives here, so this module
can be unit-tested and run locally as well as cloned into Colab. The notebook
(`notebooks/wan_loop.ipynb`) is a thin launcher that sets parameters and calls
`run()` from this module.

Local quick run:
    python src/wan_loop.py path/to/image.jpg
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from PIL import Image

# --- Default prompts -------------------------------------------------------
DEFAULT_PROMPT = (
    "A dreamy looping music visual based on the input image. Keep the exact "
    "subject, composition, colors, and background consistent. Add only gentle "
    "ambient motion: soft drifting particles, subtle light shimmer, very slow "
    "atmospheric movement, slight breathing motion in the scene. Camera locked. "
    "No zoom, no pan, no rotation. Smooth, cinematic, stable, hypnotic."
)

DEFAULT_NEGATIVE_PROMPT = (
    "camera shake, zoom, pan, rotation, fast motion, scene change, new objects, "
    "new people, flicker, glitch, face morphing, body distortion, text "
    "distortion, logo distortion, sudden lighting change, jump cut"
)


@dataclass
class Config:
    """All the knobs the notebook exposes."""

    model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    num_frames: int = 121          # clip length in frames
    fps: int = 24                  # playback frame rate
    num_inference_steps: int = 20  # more = slower, potentially cleaner
    guidance_scale: float = 5.0    # prompt adherence
    seed: int = 42                 # change for a different result
    max_long_side: int = 832       # cap the longest image side (VRAM/quality)
    output_dir: str = "outputs"


# --- Image sizing ----------------------------------------------------------
def round_to_multiple(x, base=16):
    """Round x to the nearest multiple of `base` (minimum one multiple)."""
    return max(base, int(round(x / base)) * base)


def fit_size(w, h, max_long_side=832):
    """Scale (w, h) so the longest side <= max_long_side, then snap to /16."""
    long_side = max(w, h)
    scale = min(1.0, max_long_side / float(long_side))
    return round_to_multiple(w * scale), round_to_multiple(h * scale)


def load_image(path, max_long_side=832):
    """Open an image, resize to a model-friendly generation size.

    Returns (image, (orig_w, orig_h), (gen_w, gen_h)).
    """
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    gen_w, gen_h = fit_size(orig_w, orig_h, max_long_side)
    img = img.resize((gen_w, gen_h), Image.LANCZOS)
    return img, (orig_w, orig_h), (gen_w, gen_h)


# --- Model -----------------------------------------------------------------
def load_pipeline(cfg: Config, device: str | None = None):
    """Load the Wan2.2 TI2V-5B image-to-video pipeline.

    Uses `WanImageToVideoPipeline` (NOT `WanPipeline`, which is text-to-video
    only and rejects `image`). The VAE is loaded in float32 for stability per
    the model card; the transformer runs in bfloat16. On limited Colab VRAM,
    model CPU offload is preferred over a full move to GPU.

    Requires Diffusers from git main (image-to-video for this model landed
    after the last stable release).
    """
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    vae = AutoencoderKLWan.from_pretrained(
        cfg.model_id, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        cfg.model_id, vae=vae, torch_dtype=dtype
    )

    try:
        pipe.enable_model_cpu_offload()
    except Exception as e:  # noqa: BLE001 - offload is best-effort
        print("CPU offload unavailable, moving fully to device:", e)
        pipe = pipe.to(device)

    return pipe, device


# --- Generation ------------------------------------------------------------
def generate_frames(pipe, image, cfg: Config, device: str):
    """Run the pipeline and return the list of generated frames.

    Some pipeline versions don't accept height/width; if so we retry without
    them and let the pipeline infer size from the image.
    """
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    gen_w, gen_h = image.size
    kwargs = dict(
        image=image,
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        num_frames=cfg.num_frames,
        guidance_scale=cfg.guidance_scale,
        num_inference_steps=cfg.num_inference_steps,
        generator=generator,
    )

    try:
        result = pipe(height=gen_h, width=gen_w, **kwargs)
    except TypeError as e:
        print("Pipeline rejected height/width, retrying without them:", e)
        result = pipe(**kwargs)

    return result.frames[0]


# --- Looping & export ------------------------------------------------------
def make_pingpong(frames):
    """Forward + reverse (without duplicating the endpoints) = seamless loop."""
    return frames + frames[-2:0:-1]


def export_clip(frames, path, fps):
    """Write frames to an mp4, creating the parent directory if needed."""
    from diffusers.utils import export_to_video

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    export_to_video(frames, path, fps=fps)
    return path


# --- Top-level orchestration ----------------------------------------------
def run(image_path, cfg: Config | None = None):
    """Full pipeline: load model -> generate -> build loop -> export.

    Returns {"base": <path>, "pingpong": <path>}.
    """
    cfg = cfg or Config()

    image, orig, gen = load_image(image_path, cfg.max_long_side)
    print(f"Original size:   {orig[0]} x {orig[1]}")
    print(f"Generation size: {gen[0]} x {gen[1]}")

    pipe, device = load_pipeline(cfg)
    print("Generating...")
    frames = generate_frames(pipe, image, cfg, device)
    print("Generated frames:", len(frames))

    base = export_clip(
        frames, os.path.join(cfg.output_dir, "wan_base.mp4"), cfg.fps
    )
    loop_frames = make_pingpong(frames)
    pingpong = export_clip(
        loop_frames, os.path.join(cfg.output_dir, "wan_pingpong_loop.mp4"), cfg.fps
    )

    print("Wrote:", base)
    print("Wrote:", pingpong, "(frames:", len(loop_frames), ")")
    return {"base": base, "pingpong": pingpong}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python src/wan_loop.py path/to/image.jpg")
    run(sys.argv[1])
