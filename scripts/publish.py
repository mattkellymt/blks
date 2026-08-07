#!/usr/bin/env python3
"""
CalVer Release & Publishing Script for blks.
"""

from datetime import datetime
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT_DIR = Path(__file__).parent.parent.resolve()
PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"
INIT_PATH = ROOT_DIR / "src" / "blks" / "__init__.py"


def get_version() -> str:
    """Read current version from pyproject.toml using tomllib."""
    if not PYPROJECT_PATH.exists():
        sys.exit(f"Error: File not found: {PYPROJECT_PATH}")

    with open(PYPROJECT_PATH, "rb") as f:
        data = tomllib.load(f)

    try:
        return data["project"]["version"]
    except KeyError:
        sys.exit("Error: 'project.version' field missing in pyproject.toml")


def update_version(current_version: str) -> str:
    """Calculate next YYYY.M.BUILD version based on today's date."""
    now = datetime.now()
    year = now.year
    month = now.month

    parts = current_version.split(".")
    if len(parts) == 3 and parts[0] == str(year) and parts[1] == str(month):
        next_build = int(parts[2]) + 1
    else:
        next_build = 1

    return f"{year}.{month}.{next_build}"


def update_pyproject(old_ver: str, new_ver: str) -> None:
    """Replace version string in pyproject.toml and __init__.py."""
    for path in [PYPROJECT_PATH, INIT_PATH]:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace(f'"{old_ver}"', f'"{new_ver}"'), encoding="utf-8")


def main():
    current_ver = get_version()
    new_ver = update_version(current_ver)
    tag_name = f"v{new_ver}"

    print(f"Bumping version: {current_ver} -> {new_ver}")

    # 1. Update version strings
    update_pyproject(current_ver, new_ver)

    # 2. Sync virtual environment and lockfile (upgrade dependencies)
    print("Syncing virtual environment and upgrading dependencies with uv...")
    subprocess.run(["uv", "sync", "--upgrade"], cwd=ROOT_DIR, check=True)

    # 3. Commit, tag, and push to GitHub
    subprocess.run(["git", "add", "."], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "commit", "-m", f"release: {new_ver}"], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "tag", tag_name], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "push"], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "push", "origin", tag_name], cwd=ROOT_DIR, check=True)

    print(f"Released {tag_name} to GitHub & PyPI!")


if __name__ == "__main__":
    main()
