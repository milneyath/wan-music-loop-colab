#!/usr/bin/env bash
#
# Sync the Jupytext source notebook into a real .ipynb, and optionally
# commit + push the result.
#
#   scripts/sync_notebook.sh                 # just convert
#   scripts/sync_notebook.sh --push          # convert, commit, push (default msg)
#   scripts/sync_notebook.sh --push "my msg" # convert, commit, push (custom msg)

set -euo pipefail

# --- Always run from the repo root (parent of this script's dir) ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SRC="src/wan_loop.py"
OUT="notebooks/wan_loop.ipynb"

# --- Ensure jupytext is available -----------------------------------------
if ! command -v jupytext >/dev/null 2>&1; then
  echo "ERROR: 'jupytext' not found." >&2
  echo "Install it with:" >&2
  echo "    pip install jupytext" >&2
  echo "  or:" >&2
  echo "    make install" >&2
  exit 1
fi

# --- Convert percent-script -> notebook -----------------------------------
echo "Converting ${SRC} -> ${OUT}"
mkdir -p "$(dirname "${OUT}")"
jupytext --to ipynb "${SRC}" --output "${OUT}"
echo "Done."

# --- Optional commit + push ------------------------------------------------
if [[ "${1:-}" == "--push" ]]; then
  COMMIT_MSG="${2:-update notebook}"

  git add "${SRC}" "${OUT}"

  # Don't fail when there's nothing staged to commit.
  if git diff --cached --quiet; then
    echo "Nothing to commit; working tree clean."
  else
    git commit -m "${COMMIT_MSG}"
  fi

  echo "Pushing to origin main..."
  git push origin main
fi
