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

# Backend: "ltx" (Lightricks LTX-Video — robust, works on stable diffusers,
# good image-to-video) or "wan" (Wan2.2 TI2V-5B — needs diffusers git main).
BACKEND = "ltx"
MODEL_ID = None  # None -> the backend's default model

# Describe the MOTION you want — LTX only moves if you ask it to. Putting
# "static/still" in the negative prompt (not the positive) encourages motion.
PROMPT = """A dreamy, cinematic looping music visual based on the input image. The scene comes alive with gentle, continuous motion: soft drifting particles, flowing light and shimmering reflections, slowly swaying elements, and a very slow, smooth camera push-in. Everything breathes and moves naturally while keeping the subject and composition. Hypnotic, atmospheric, high quality."""

NEGATIVE_PROMPT = """static, still, frozen, motionless, no motion, blurry, jittery, distorted, deformed, flickering, glitch, low quality, worst quality, jpeg artifacts, watermark, text, scene change, jump cut"""

NUM_FRAMES = 121          # clip length in frames (must be 8*N + 1 for LTX)
FPS = 24                  # playback frame rate
# NOTE: the default "ltx" backend is the 0.9.7 *distilled* model — it ignores
# the three knobs below and uses fixed distilled settings (guidance 1.0 + the
# documented 8 custom timesteps). They apply only to the "wan" backend.
NUM_INFERENCE_STEPS = 50  # Wan only
GUIDANCE_SCALE = 5.0      # Wan only (~5 for Wan)
FLOW_SHIFT = None         # Wan only: UniPC flow shift (None = auto)
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
# The default "ltx" backend works on *stable* diffusers, so this is a plain,
# safe install: no `-U`, no git main. Because Colab already ships a recent
# diffusers, the `>=` floor is normally already satisfied and pip changes
# nothing — so numpy is left intact and `import torch` (in the Run cell) can't
# segfault. We also never import torch in this cell.
# (The "wan" backend additionally needs diffusers git main — see README.)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "diffusers>=0.32", "ftfy", "imageio-ffmpeg"],
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
