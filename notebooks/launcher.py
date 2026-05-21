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
# Turns a **single still image** into a short image-to-video clip, then builds
# a seamless **ping-pong loop** — a hypnotic background visual for a music
# track. Loop the result under your song later in a video editor.
#
# **All the real code lives in [`src/wan_loop.py`](https://github.com/milneyath/wan-music-loop-colab/blob/main/src/wan_loop.py)** in the repo; this notebook just
# clones it, installs deps, and runs it.
#
# **How to use:**
# 1. *Runtime → Change runtime type → **GPU*** (A100 recommended for `wan`).
# 2. Edit the **Parameters** cell.
# 3. *Runtime → **Run all***. If Colab shows a **RESTART** button after Setup,
#    click it, then run the **Run** cell. Upload an image when prompted; the
#    clips download automatically when done.

# %% [markdown]
# ## A. Parameters — edit these

# %%
# Repo to clone (change if you forked it under a different account).
REPO_URL = "https://github.com/milneyath/wan-music-loop-colab.git"

# Backend: "wan" (Wan2.2 TI2V-5B, best quality, needs an A100) or
# "ltx" (Lightricks LTX-Video, lighter, fits a free T4).
BACKEND = "wan"
MODEL_ID = None  # None -> the backend's default model

PROMPT = """A dreamy looping music visual based on the input image. Keep the exact subject, composition, colors, and background consistent. Add only gentle ambient motion: soft drifting particles, subtle light shimmer, very slow atmospheric movement, slight breathing motion in the scene. Camera locked. No zoom, no pan, no rotation. Smooth, cinematic, stable, hypnotic."""

NEGATIVE_PROMPT = """camera shake, zoom, pan, rotation, fast motion, scene change, new objects, new people, flicker, glitch, face morphing, body distortion, text distortion, logo distortion, sudden lighting change, jump cut"""

NUM_FRAMES = 121          # clip length in frames
FPS = 24                  # playback frame rate
NUM_INFERENCE_STEPS = 40  # Wan needs >=40; fewer leaves gray static noise
GUIDANCE_SCALE = 5.0      # prompt adherence
FLOW_SHIFT = None         # UniPC flow shift; None = auto (5.0 @720p, 3.0 @480p)
SEED = 42                 # change for a different result
MAX_LONG_SIDE = 832       # cap the longest image side (raise toward 1280 for
                          # 720p quality on an A100; lower it if you OOM)

# %% [markdown]
# ## B. Setup — clone the repo and install dependencies
#
# This clones/updates the repo and installs a **stable** Diffusers release.
# It deliberately does **not** upgrade or import torch, so it can't crash the
# kernel. **If Colab shows a RESTART button when this finishes, click it, then
# run the next cell.**

# %%
import os
import subprocess
import sys

# --- Clone or update the repo ---------------------------------------------
REPO_DIR = REPO_URL.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)

# --- Install dependencies --------------------------------------------------
# Wan2.2 TI2V-5B *image-to-video* only decodes correctly on diffusers git main
# (the expand_timesteps I2V path); the latest *stable* (<=0.38) renders washed-
# out gray. So install diffusers from main — but with `--no-deps` so pip
# touches ONLY diffusers and does NOT upgrade numpy/torch/etc. Colab already
# has all of diffusers' runtime deps; upgrading numpy out from under the
# resident torch is exactly what segfaults `import torch` ("Runtime
# disconnected"). ftfy + imageio-ffmpeg are small and pull nothing heavy.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
     "git+https://github.com/huggingface/diffusers"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ftfy", "imageio-ffmpeg"],
    check=True,
)

print("Setup done. If Colab shows a RESTART button, click it, then run the next cell.")

# %% [markdown]
# ## C. Run — upload an image and generate the loop

# %%
import os
import sys

sys.path.insert(0, os.path.join(os.path.abspath(REPO_DIR), "src"))
import wan_loop  # noqa: E402

cfg = wan_loop.Config(
    backend=BACKEND,
    model_id=MODEL_ID,
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_frames=NUM_FRAMES,
    fps=FPS,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    flow_shift=FLOW_SHIFT,
    seed=SEED,
    max_long_side=MAX_LONG_SIDE,
)
wan_loop.colab_run(cfg)
