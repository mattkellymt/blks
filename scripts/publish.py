#!/usr/bin/env python3
"""
CalVer Release & Publishing Script for blks.
"""

from datetime import datetime
from pathlib import Path
import subprocess

ROOT_DIR = Path(__file__).parent.parent.resolve()
PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"
INIT_PATH = ROOT_DIR / "src" / "blks" / "__init__.py"


def get_current_version() -> str:
    """Read the current version from pyproject.toml."""
    for line in PYPROJECT_PATH.read_text().splitlines():
        if line.startswith("version ="):
            return line.split('"')[1]
    return "0.1.0"


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


def update_file(path: Path, old_ver: str, new_ver: str) -> None:
    """Replace old version string with new version string in a file."""
    if path.exists():
        text = path.read_text()
        path.write_text(text.replace(f'"{old_ver}"', f'"{new_ver}"'))


def main():
    current_ver = get_current_version()
    new_ver = calculate_next_calver(current_ver)
    tag_name = f"v{new_ver}"

    print(f"Bumping version: {current_ver} -> {new_ver}")

    # 1. Update pyproject.toml & __init__.py
    update_file(PYPROJECT_PATH, current_ver, new_ver)
    update_file(INIT_PATH, current_ver, new_ver)

    # 2. Commit, tag, and push to GitHub
    subprocess.run(["git", "add", "."], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "commit", "-m", f"release: {new_ver}"], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "tag", tag_name], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "push"], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "push", "origin", tag_name], cwd=ROOT_DIR, check=True)

    print(f"Released {tag_name} to GitHub & PyPI!")


if __name__ == "__main__":
    main()
