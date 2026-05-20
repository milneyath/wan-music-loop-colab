#!/usr/bin/env python3
"""Print the Google Colab URL for this repo's notebook.

Reads GITHUB_USERNAME and REPO_NAME from the environment when available,
falling back to placeholder defaults.
"""

import os


def main() -> None:
    username = os.environ.get("GITHUB_USERNAME", "YOUR_GITHUB_USERNAME")
    repo = os.environ.get("REPO_NAME", "wan-music-loop-colab")
    url = (
        f"https://colab.research.google.com/github/{username}/{repo}"
        "/blob/main/notebooks/wan_loop.ipynb"
    )
    print(url)


if __name__ == "__main__":
    main()
