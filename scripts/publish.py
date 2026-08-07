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


def get_current_version() -> str:
    """Read current version from pyproject.toml using tomllib."""
    if not PYPROJECT_PATH.exists():
        sys.exit(f"Error: File not found: {PYPROJECT_PATH}")

    with open(PYPROJECT_PATH, "rb") as f:
        data = tomllib.load(f)

    try:
        return data["project"]["version"]
    except KeyError:
        sys.exit("Error: 'project.version' field missing in pyproject.toml")


def calculate_next_calver(current_version: str) -> str:
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


def update_pyproject(new_version: str) -> None:
    """Update version line in pyproject.toml."""
    lines = []
    for line in PYPROJECT_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("version ="):
            lines.append(f'version = "{new_version}"')
        else:
            lines.append(line)
    PYPROJECT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_init(new_version: str) -> None:
    """Update __version__ line in src/blks/__init__.py."""
    if not INIT_PATH.exists():
        return
    lines = []
    for line in INIT_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__ ="):
            lines.append(f'__version__ = "{new_version}"')
        else:
            lines.append(line)
    INIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    current_ver = get_current_version()
    new_ver = calculate_next_calver(current_ver)
    tag_name = f"v{new_ver}"

    print(f"Bumping version: {current_ver} -> {new_ver}")

    # 1. Update pyproject.toml & __init__.py
    update_pyproject(new_ver)
    update_init(new_ver)

    # 2. Commit, tag, and push to GitHub
    subprocess.run(["git", "add", "."], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "commit", "-m", f"release: {new_ver}"], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "tag", tag_name], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "push"], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "push", "origin", tag_name], cwd=ROOT_DIR, check=True)

    print(f"Released {tag_name} to GitHub & PyPI!")


if __name__ == "__main__":
    main()
