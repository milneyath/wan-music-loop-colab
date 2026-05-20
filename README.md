# wan-music-loop-colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/wan-music-loop-colab/blob/main/notebooks/wan_loop.ipynb)

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

> ⚠️ **Replace `YOUR_GITHUB_USERNAME`** in the badge URL (and in `.env`) with
> your actual GitHub username *after* you create the GitHub repo — otherwise the
> Colab button won't resolve to your notebook.

---

## Why Jupytext?

Notebooks (`.ipynb`) are JSON and awful to diff and edit in a normal editor.
With Jupytext you keep the **source of truth as a Python script**
(`src/wan_loop.py`, in "percent" cell format), edit it comfortably in Cursor or
VS Code, and generate the `.ipynb` only when you want to run it in Colab.

---

## Local setup

```bash
git clone git@github.com:YOUR_GITHUB_USERNAME/wan-music-loop-colab.git
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

## Editing in Cursor / VS Code

Open the repo and edit **`src/wan_loop.py`**. It is a Jupytext *percent*
notebook — cells are delimited by `# %%` (code) and `# %% [markdown]`
(markdown). Edit it like any Python file; no notebook UI required.

Do **not** hand-edit `notebooks/wan_loop.ipynb` — it is generated.

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
jupytext --to ipynb src/wan_loop.py --output notebooks/wan_loop.ipynb
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
https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/wan-music-loop-colab/blob/main/notebooks/wan_loop.ipynb
```

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

---

## In Colab

1. Set the runtime to **GPU**: *Runtime → Change runtime type → GPU*.
2. Run the cells top to bottom.
3. Upload a still image when prompted.
4. The notebook generates the base clip and a seamless ping-pong loop (one
   run-through), then downloads them.

> **VRAM note:** the larger **Wan2.2 I2V A14B** models can need very high VRAM
> and generally won't fit on a free Colab GPU. This notebook defaults to
> **`Wan-AI/Wan2.2-TI2V-5B-Diffusers`**, the practical Colab starting point.

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
    sync_notebook.sh      # src/*.py -> notebooks/*.ipynb (+ optional push)
    open_colab_url.py      # prints the Colab URL from env vars
  src/
    wan_loop.py            # EDIT THIS — Jupytext percent notebook
  notebooks/
    wan_loop.ipynb         # GENERATED — opened in Colab
```

Model weights and output videos are **not** committed (see `.gitignore`).
