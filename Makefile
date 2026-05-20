# wan-music-loop-colab — developer tasks
#
# Usage:
#   make install            Install Python dependencies
#   make notebook           Sync src/wan_loop.py -> notebooks/wan_loop.ipynb
#   make push msg="..."     Sync, commit, and push to origin main
#   make colab              Print the Colab URL for the notebook

.PHONY: install notebook push colab

# Load GITHUB_USERNAME / REPO_NAME from .env if present (optional).
-include .env
export

install:
	pip install -r requirements.txt

notebook:
	bash scripts/sync_notebook.sh

# Pass a message:  make push msg="my change"
msg ?= update notebook
push:
	bash scripts/sync_notebook.sh --push "$(msg)"

colab:
	python scripts/open_colab_url.py
