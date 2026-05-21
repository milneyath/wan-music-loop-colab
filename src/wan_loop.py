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


# --- Backend registry ------------------------------------------------------
# Each backend maps to (pipeline class name, default model id). Both pipelines
# accept `image` + `prompt`, so the rest of the engine stays backend-agnostic.
# Class names are resolved lazily inside load_pipeline so this module imports
# without diffusers/torch present.
BACKENDS = {
    "wan": ("WanImageToVideoPipeline", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
    "ltx": ("LTXImageToVideoPipeline", "Lightricks/LTX-Video"),
}


@dataclass
class Config:
    """All the knobs the notebook exposes."""

    backend: str = "wan"           # "wan" (default, A100) or "ltx" (lighter)
    model_id: str | None = None    # None -> backend default from BACKENDS
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    num_frames: int = 121          # clip length in frames
    fps: int = 24                  # playback frame rate
    num_inference_steps: int = 40  # more = slower, cleaner (Wan wants >=40)
    guidance_scale: float = 5.0    # prompt adherence
    flow_shift: float | None = None  # UniPC flow shift; None -> auto by res
    seed: int = 42                 # change for a different result
    max_long_side: int = 832       # cap the longest image side (VRAM/quality)
    output_dir: str = "outputs"


# --- Image sizing ----------------------------------------------------------
def round_to_multiple(x, base=16):
    """Round x to the nearest multiple of `base` (minimum one multiple)."""
    return max(base, int(round(x / base)) * base)


# Wan2.2 needs generation dims that are a multiple of
# vae_scale_factor_spatial (16) * transformer patch_size (2) = 32.
WAN_MOD = 32


def fit_size(w, h, max_long_side=832, base=WAN_MOD):
    """Scale (w, h) so the longest side <= max_long_side, then snap to `base`."""
    long_side = max(w, h)
    scale = min(1.0, max_long_side / float(long_side))
    return round_to_multiple(w * scale, base), round_to_multiple(h * scale, base)


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
def resolve_backend(cfg: Config):
    """Return (pipeline_class_name, model_id) for the configured backend."""
    if cfg.backend not in BACKENDS:
        raise ValueError(
            f"Unknown backend {cfg.backend!r}; choose from {list(BACKENDS)}"
        )
    cls_name, default_model = BACKENDS[cfg.backend]
    return cls_name, (cfg.model_id or default_model)


# A100-class cards have ~40GB; anything at/above this gets the fast no-offload
# path. Below it (e.g. a free T4 with 16GB) we fall back to CPU offload.
HIGH_VRAM_BYTES = 30 * 1024**3

# A loaded pipeline is cached here keyed by (cls_name, model_id) so re-running
# the Run cell in the same Colab session reuses the resident model instead of
# loading a *second* copy onto the GPU. The latter OOMs on a 40GB A100 because
# IPython keeps the previous run's pipeline alive via its stored traceback.
_PIPELINE_CACHE: dict = {}


def free_pipelines():
    """Drop any cached pipeline and reclaim its GPU memory."""
    import gc

    _PIPELINE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_pipeline(cfg: Config, device: str | None = None):
    """Load the configured image-to-video pipeline, VRAM-aware.

    Backend is selected via `cfg.backend` (see BACKENDS):
      - "wan" -> WanImageToVideoPipeline (Wan2.2 TI2V-5B). NOT `WanPipeline`,
        which is text-to-video only and rejects `image`.
      - "ltx" -> LTXImageToVideoPipeline (lighter, fits a free T4).

    On a high-VRAM GPU (A100, >= ~30GB) the pipeline is moved fully to CUDA for
    speed. On a smaller GPU we use `enable_model_cpu_offload()` plus VAE
    slicing. VAE *tiling* is deliberately NOT enabled — it is broken for Wan2.2
    in diffusers (huggingface/diffusers#12529).

    The result is cached so a second call in the same session (e.g. re-running
    the Run cell) reuses the resident model rather than loading another copy.
    """
    import diffusers

    cls_name, model_id = resolve_backend(cfg)
    key = (cls_name, model_id)

    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        print("Reusing already-loaded pipeline (cached this session).")
        return cached

    # A different model was requested, or a prior run left one resident — free
    # it before loading so we don't stack two pipelines on the GPU.
    free_pipelines()

    PipelineClass = getattr(diffusers, cls_name)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Wan2.2 wants its VAE in float32 for stability (per the model card).
    if cfg.backend == "wan":
        vae = diffusers.AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = PipelineClass.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
    else:
        pipe = PipelineClass.from_pretrained(model_id, torch_dtype=dtype)

    total_vram = (
        torch.cuda.get_device_properties(0).total_memory
        if device == "cuda"
        else 0
    )

    if device == "cuda" and total_vram >= HIGH_VRAM_BYTES:
        print(f"High-VRAM GPU ({total_vram / 1024**3:.0f}GB): full move to CUDA.")
        pipe = pipe.to("cuda")
    elif device == "cuda":
        print(
            f"Limited-VRAM GPU ({total_vram / 1024**3:.0f}GB): CPU offload + "
            "VAE slicing."
        )
        pipe.enable_model_cpu_offload()
        try:
            pipe.vae.enable_slicing()
        except Exception as e:  # noqa: BLE001 - slicing is best-effort
            print("VAE slicing unavailable:", e)
    else:
        pipe = pipe.to(device)

    _PIPELINE_CACHE[key] = (pipe, device)
    return pipe, device


# --- Generation ------------------------------------------------------------
def auto_flow_shift(gen_w, gen_h):
    """Pick a UniPC flow shift by resolution: 5.0 for ~720p, 3.0 for ~480p.

    Diffusers' Wan docs recommend these tiers; using the 720p shift at a low
    resolution (or leaving the model default) under-denoises and yields gray /
    static-noise output.
    """
    return 5.0 if gen_w * gen_h >= (1280 * 704) // 2 else 3.0


def configure_scheduler(pipe, cfg: Config, gen_w, gen_h):
    """Swap in UniPCMultistepScheduler with the right flow shift (Wan only).

    Returns the flow shift used, or None if left unchanged (e.g. non-Wan
    backends, or if the scheduler can't be reconfigured).
    """
    if cfg.backend != "wan":
        return None
    shift = cfg.flow_shift if cfg.flow_shift is not None else auto_flow_shift(gen_w, gen_h)
    try:
        from diffusers import UniPCMultistepScheduler

        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=shift
        )
        return shift
    except Exception as e:  # noqa: BLE001 - fall back to the shipped scheduler
        print("Could not set UniPC scheduler, using default:", e)
        return None


def generate_frames(pipe, image, cfg: Config, device: str):
    """Run the pipeline and return the list of generated frames.

    Some pipeline versions don't accept height/width; if so we retry without
    them and let the pipeline infer size from the image.
    """
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    gen_w, gen_h = image.size
    shift = configure_scheduler(pipe, cfg, gen_w, gen_h)
    if shift is not None:
        print(f"Scheduler: UniPCMultistepScheduler (flow_shift={shift})")
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


def describe_frames(frames):
    """Print min/max/mean/std of the generated frames.

    A near-constant array (std ~0, mean ~0.5) means the decode collapsed to
    flat gray — useful for telling a real result apart from a broken one.
    """
    import numpy as np

    arr = np.asarray(frames, dtype=np.float32)
    flat = arr.reshape(-1)
    print(
        f"Frame stats: shape={arr.shape} dtype~{arr.dtype} "
        f"min={flat.min():.4f} max={flat.max():.4f} "
        f"mean={flat.mean():.4f} std={flat.std():.4f}"
    )
    if flat.std() < 1e-3:
        print("  WARNING: frames are essentially uniform — output is flat gray.")


# --- Looping & export ------------------------------------------------------
def make_pingpong(frames):
    """Forward + reverse (without duplicating the endpoints) = seamless loop.

    `frames` may be a list or a numpy array (depending on the diffusers
    version). Coerce to a list of per-frame items first so the `+` is list
    concatenation, not numpy elementwise addition.
    """
    forward = list(frames)
    return forward + forward[-2:0:-1]


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
    describe_frames(frames)

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


# --- Colab entry point -----------------------------------------------------
def colab_run(cfg: Config | None = None):
    """Colab front door: detect GPU, upload an image, run, download outputs.

    All `google.colab` imports live inside this function so the module stays
    importable locally and in plain Jupyter. Call this from the notebook's
    Run cell after building a Config.
    """
    cfg = cfg or Config()

    import diffusers

    print("diffusers version:", diffusers.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1024**3:.0f}GB)")
    else:
        print("WARNING: no GPU. Set Runtime -> Change runtime type -> GPU.")

    _, model_id = resolve_backend(cfg)
    print(f"Backend: {cfg.backend}  Model: {model_id}")

    from google.colab import files

    print("Upload a still image (the first file is used):")
    uploaded = files.upload()
    image_path = next(iter(uploaded))
    print("Using:", image_path)

    paths = run(image_path, cfg)
    files.download(paths["base"])
    files.download(paths["pingpong"])
    return paths


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python src/wan_loop.py path/to/image.jpg")
    run(sys.argv[1])
