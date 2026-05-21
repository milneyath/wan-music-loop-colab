# wan-music-loop-colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/milneyath/wan-music-loop-colab/blob/main/notebooks/wan_loop.ipynb)

Generate short, **seamlessly looping music visuals** from a single still image
using [**Wan2.2**](https://huggingface.co/Wan-AI) via 🤗 **Diffusers**, all in
Google Colab — while you edit the code locally in **Cursor** or **VS Code**.

The workflow:

1. Edit the notebook **as a plain Python file** (`src/wan_loop.py`) locally.
2. Sync it into a real Jupyter notebook (`notebooks/wan_loop.ipynb`) with
   [Jupytext](https://jupytext.readthedocs.io/).
3. Push to GitHub.
4. Open it in **Google Colab** with one click via the badge above.

In Colab the notebook generates a short Wan2.2 image-to-video clip, then builds
a seamless **ping-pong loop** and exports a single run-through. Loop that clip
under your track later in a video editor rather than exporting a giant file
here.

> This repo is configured for the GitHub user **`milneyath`**. If you fork it
> under a different account, update the username in the badge URL above and in
> your `.env` so the Colab button resolves to your copy.

---

## Why Jupytext?

Notebooks (`.ipynb`) are JSON and awful to diff and edit in a normal editor.
With Jupytext you keep the **source of truth as a Python script**
(`src/wan_loop.py`, in "percent" cell format), edit it comfortably in Cursor or
VS Code, and generate the `.ipynb` only when you want to run it in Colab.

---

## Local setup

```bash
git clone git@github.com:milneyath/wan-music-loop-colab.git
cd wan-music-loop-colab

# (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# install tooling (jupytext + the libraries the notebook uses)
make install
# or: pip install -r requirements.txt
```

Copy the example env file and fill in your details:

```bash
cp .env.example .env
# then edit .env:
#   GITHUB_USERNAME=your-actual-username
#   REPO_NAME=wan-music-loop-colab
```

### Installing Jupytext on its own

If you only want the sync tool:

```bash
pip install jupytext
```

---

## How the code is organised

The notebook is deliberately **thin** — it sets parameters and then clones,
installs, and runs. All real logic lives in an importable module:

- **`src/wan_loop.py`** — the engine: `Config` plus `load_pipeline`,
  `generate_frames`, `make_pingpong`, `export_clip`, a top-level `run()`, and a
  `colab_run()` entry point (the only place `google.colab` is imported). Plain
  Python with no Colab dependency at import time, so you can test it locally:
  `python src/wan_loop.py path/to/image.jpg`.
- **`notebooks/launcher.py`** — the Jupytext *percent* source for the notebook.
  **Three thin cells**: **A. Parameters** (what you tweak day to day),
  **B. Setup** (clone + install — no torch import), and **C. Run** (~5 lines:
  import the engine and call `wan_loop.colab_run(cfg)`).
- **`notebooks/wan_loop.ipynb`** — generated from `launcher.py`. Don't
  hand-edit it.

The Setup cell does `git clone` / `git pull`, so to pick up engine changes you
just **re-run** it (it pulls the latest `main`) — no notebook re-sync needed for
logic edits.

---

## Sync the notebook

Convert the Python source into a runnable `.ipynb`:

```bash
make notebook
# or directly:
bash scripts/sync_notebook.sh
```

This runs:

```bash
jupytext --to ipynb notebooks/launcher.py --output notebooks/wan_loop.ipynb
```

---

## Push to GitHub

Sync, commit, and push in one step:

```bash
make push msg="describe your change"
# or directly:
bash scripts/sync_notebook.sh --push "describe your change"
```

The script commits both `src/wan_loop.py` and `notebooks/wan_loop.ipynb`, then
pushes to `origin main`. It won't fail if there's nothing to commit.

---

## Use the Colab button

Once the notebook is on GitHub's `main` branch, click the **Open In Colab**
badge at the top of this README. Colab loads the notebook straight from GitHub.

To print the URL for your configured username/repo:

```bash
make colab
# or:
python scripts/open_colab_url.py
```

### Colab URL template

```
https://colab.research.google.com/github/milneyath/wan-music-loop-colab/blob/main/notebooks/wan_loop.ipynb
```

Replace `milneyath` with your GitHub username if you forked under a different
account.

---

## In Colab

1. Set the runtime to **GPU**: *Runtime → Change runtime type → GPU*.
2. Edit the **A. Parameters** cell (backend, prompt, frames, resolution, …).
3. *Runtime → Run all.* **B. Setup** clones the repo and installs Diffusers
   from git main. **If Colab shows a RESTART button when Setup finishes, click
   it, then run the cells again** — this is normal after a dependency install.
4. **C. Run** imports the engine, prompts you to upload a still image, then
   generates the base clip and a seamless ping-pong loop and downloads them.

> **Why Diffusers from git main?** Wan2.2 TI2V-5B *image-to-video* only decodes
> correctly on Diffusers main (the `expand_timesteps` I2V path); the latest
> stable (≤0.38) renders washed-out gray. Setup installs it from main.
>
> **Why the restart-and-rerun?** "Runtime disconnected" happens when something
> upgrades **numpy** out from under Colab's resident **torch**: the next
> `import torch` then segfaults and kills the kernel. The first version did
> this via `-U`; installing diffusers-from-main *with its deps* did it too
> (pip bumped numpy). Setup now installs diffusers main with **`--no-deps`** so
> pip touches only diffusers, and **never imports torch** in that cell — so the
> import happens in the Run cell (a clean session) and the kernel survives.
>
> ⚠️ If you previously ran a version that upgraded numpy, *Restart session*
> won't undo it — use **Runtime → Disconnect and delete runtime** to get a
> fresh VM, then reopen the notebook and Run all.

### Backend & GPU guidance

The `BACKEND` parameter switches the model:

- **`"ltx"`** (default) → `Lightricks/LTX-Video-0.9.7-distilled` via
  `LTXConditionPipeline`. The variant the authors built for strong
  image-to-video motion; guidance-distilled, so the engine runs it at
  `guidance_scale=1.0` with the documented 8 custom timesteps (single-stage, no
  latent upsampler). Works on **stable** diffusers. The 13B distilled model is
  large but fits an A100; on smaller GPUs the engine uses CPU offload + VAE
  slicing.
- **`"wan"`** → `Wan-AI/Wan2.2-TI2V-5B-Diffusers`. Higher fidelity but fussy:
  its image-to-video path is **only** in diffusers **git main** (stable
  releases decode to washed-out gray), and it wants an A100. To use it, set
  `BACKEND="wan"` *and* change the Setup install to diffusers from main
  (`pip install --no-deps "git+https://github.com/huggingface/diffusers"` — the
  `--no-deps` matters, otherwise pip bumps numpy and the kernel crashes).

Generation dimensions are snapped to a multiple of **32**, which both backends
require. For the `"wan"` backend the engine also swaps in
`UniPCMultistepScheduler` with a resolution-aware `flow_shift` and leaves VAE
*tiling* off (broken in diffusers,
[#12529](https://github.com/huggingface/diffusers/issues/12529)).

---

## Repo layout

```
wan-music-loop-colab/
  README.md
  requirements.txt
  .gitignore
  .env.example
  Makefile
  scripts/
    sync_notebook.sh       # launcher.py -> wan_loop.ipynb (+ optional push)
    open_colab_url.py      # prints the Colab URL from env vars
  src/
    wan_loop.py            # ENGINE — importable functions + Config (edit logic here)
  notebooks/
    launcher.py            # Jupytext source for the thin notebook (edit params here)
    wan_loop.ipynb         # GENERATED — opened in Colab
```

Model weights and output videos are **not** committed (see `.gitignore`).
