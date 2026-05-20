# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Wan2.2 Music Loop Generator
#
# This notebook turns a **single input image** into a short
# **Wan2.2 image-to-video** clip, then builds a seamless **ping-pong loop**
# that you can extend to song length — ideal as a hypnotic background visual
# for a YouTube music track.
#
# **Pipeline:**
# 1. Upload a still image.
# 2. Generate gentle ambient motion with `Wan-AI/Wan2.2-TI2V-5B-Diffusers`.
# 3. Export the base clip.
# 4. Build a ping-pong (forward + reverse) loop so it repeats seamlessly.
#    Export one run-through — loop it under your track later in an editor.
#
# > Run on a **GPU** runtime: *Runtime → Change runtime type → GPU*.

# %% [markdown]
# ## A. Install dependencies
#
# > **Important:** Wan2.2 TI2V-5B **image-to-video** support landed after the
# > last stable Diffusers release, so we install Diffusers from **git `main`**.
# > The stable PyPI build raises `unexpected keyword argument 'image'`.

# %%
# !pip -q install -U git+https://github.com/huggingface/diffusers transformers accelerate sentencepiece safetensors imageio imageio-ffmpeg pillow moviepy

# %% [markdown]
# ## B. GPU check
#
# Make sure a GPU is attached. If this errors or shows nothing, switch the
# runtime to GPU first.

# %%
# !nvidia-smi

# %% [markdown]
# ## C. Imports

# %%
import os

import torch
from PIL import Image
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

# google.colab is only available inside Colab; guard so the file still imports
# locally (e.g. when syncing with jupytext).
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    files = None
    IN_COLAB = False

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# %% [markdown]
# ## D. User settings
#
# Tweak these to control the look and length of the clip.

# %%
MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

PROMPT = """A dreamy looping music visual based on the input image. Keep the exact subject, composition, colors, and background consistent. Add only gentle ambient motion: soft drifting particles, subtle light shimmer, very slow atmospheric movement, slight breathing motion in the scene. Camera locked. No zoom, no pan, no rotation. Smooth, cinematic, stable, hypnotic."""

NEGATIVE_PROMPT = """camera shake, zoom, pan, rotation, fast motion, scene change, new objects, new people, flicker, glitch, face morphing, body distortion, text distortion, logo distortion, sudden lighting change, jump cut"""

NUM_FRAMES = 121          # clip length in frames (model-dependent)
FPS = 24                  # playback frame rate
NUM_INFERENCE_STEPS = 20  # more steps = slower, potentially cleaner
GUIDANCE_SCALE = 5.0      # prompt adherence
SEED = 42                 # change for a different result
MAX_LONG_SIDE = 832       # cap the longest image dimension (VRAM/quality)

# %% [markdown]
# ## E. Upload an image
#
# Pick a single still image. The first uploaded file is used.

# %%
if IN_COLAB:
    uploaded = files.upload()
    image_path = next(iter(uploaded))
    print("Using:", image_path)
else:
    # When running locally, set this to a path on disk.
    image_path = "input.jpg"
    print("Not in Colab — expecting local file:", image_path)

# %% [markdown]
# ## F. Resize helpers
#
# Wan models prefer dimensions that are multiples of 16. We cap the longest
# side and round both dimensions so the pipeline is happy.

# %%
def round_to_multiple(x, base=16):
    """Round x to the nearest multiple of `base` (minimum one multiple)."""
    return max(base, int(round(x / base)) * base)


def fit_size(w, h, max_long_side=832):
    """Scale (w, h) so the longest side <= max_long_side, then snap to /16."""
    long_side = max(w, h)
    scale = min(1.0, max_long_side / float(long_side))
    new_w = round_to_multiple(w * scale)
    new_h = round_to_multiple(h * scale)
    return new_w, new_h


# Load and inspect the image.
input_image = Image.open(image_path).convert("RGB")
orig_w, orig_h = input_image.size
width, height = fit_size(orig_w, orig_h, MAX_LONG_SIDE)

# Resize to the generation resolution.
input_image = input_image.resize((width, height), Image.LANCZOS)

print(f"Original size:   {orig_w} x {orig_h}")
print(f"Generation size: {width} x {height}")

# %% [markdown]
# ## G. Load the model
#
# Wan2.2 TI2V-5B is a **unified** model: the same `WanPipeline` does both
# text-to-video and image-to-video — passing an `image` to the call switches it
# into I2V mode. The VAE is loaded separately in **float32** for stability
# (per the model card), while the transformer runs in **bfloat16**.
#
# > **VRAM warning:** the larger **Wan2.2 I2V A14B** models can require very
# > high VRAM and generally won't fit on a free Colab GPU. The
# > **TI2V-5B** model used here is the practical starting point for Colab.

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# VAE in float32, transformer/pipeline in bfloat16.
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=dtype)

# On limited Colab VRAM, model CPU offload is safer than a full move to GPU.
# (Don't also call .to("cuda") when offload is active — pick one.)
try:
    pipe.enable_model_cpu_offload()
except Exception as e:
    print("CPU offload unavailable, moving fully to device:", e)
    pipe = pipe.to(device)

generator = torch.Generator(device=device).manual_seed(SEED)

# %% [markdown]
# ## H. Generate the clip
#
# Some pipeline versions don't accept `height`/`width`; if that happens we
# retry without them and let the pipeline infer the size from the image.

# %%
common_kwargs = dict(
    image=input_image,
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_frames=NUM_FRAMES,
    guidance_scale=GUIDANCE_SCALE,
    num_inference_steps=NUM_INFERENCE_STEPS,
    generator=generator,
)

try:
    result = pipe(height=height, width=width, **common_kwargs)
except TypeError as e:
    print("Pipeline rejected height/width, retrying without them:", e)
    result = pipe(**common_kwargs)

frames = result.frames[0]
print("Generated frames:", len(frames))

# %% [markdown]
# ## I. Export the base video

# %%
os.makedirs("outputs", exist_ok=True)

base_path = "outputs/wan_base.mp4"
export_to_video(frames, base_path, fps=FPS)
print("Wrote:", base_path)

# %% [markdown]
# ## J. Create a ping-pong loop
#
# Append the clip played in reverse (skipping the duplicate end/start frames)
# so the motion eases back to where it began — a seamless loop.

# %%
loop_frames = frames + frames[-2:0:-1]

pingpong_path = "outputs/wan_pingpong_loop.mp4"
export_to_video(loop_frames, pingpong_path, fps=FPS)
print("Wrote:", pingpong_path, "frames:", len(loop_frames))

# %% [markdown]
# ## K. Download the results
#
# This exports a single run-through: the base clip and the seamless ping-pong
# loop (one pass). Loop the ping-pong file as many times as you like later in
# your video editor under the music.

# %%
if IN_COLAB:
    files.download(base_path)
    files.download(pingpong_path)
else:
    print("Not in Colab — files are in the outputs/ directory:")
    print(" ", base_path)
    print(" ", pingpong_path)
